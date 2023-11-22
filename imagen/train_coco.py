import numpy as np
import random
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torch.utils.data import Dataset
import os
import wandb

from typing import Optional, Callable, Any, List, Tuple

from dataset_coco import prepare_data

WANDB_PROJECT_NAME = "imagen"
EXPERIMENT_NAME = "coco-test-3"
TAGS = ["coco", "imagen", "original"]

MODEL_SAVE_DIR = "coco_checkpoints/"

SEED = 0

HYPERPARAMS = {
    "steps": 200_000,  # Total number of training steps (iterations) to run.
    "dim": 128,  # Base dimensionality of the model, often related to the complexity and capacity of the model.
    "cond_dim": 128,  # Dimensionality of the conditional input, e.g., for conditional GANs or conditional image generation.
    "dim_mults": (
        1,
        2,
        4,
    ),  # Multipliers for the dimensions at different stages in the network, controlling the depth and width of the model.
    "image_sizes": 256,  # The size (height and width) of the images that the model will process and generate.
    "timesteps": 250,
    "cond_drop_prob": 0.1,  # Dropout probability for the conditional inputs, used for regularization.
    "batch_size": 64,
    "lr": 1e-4,
    "num_resnet_blocks": 3,
    "model_save_dir": MODEL_SAVE_DIR,
    "dynamic_thresholding": True,
}

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)

wandb.login(key=os.environ["WANDB_API_KEY"])


class HFDataset(Dataset):
    """
    A custom dataset class for handling data from Hugging Face datasets along with precomputed embeddings.

    This dataset class is designed to work with datasets from the Hugging Face library. It expects
    the dataset to have images and labels, and it also requires precomputed embeddings corresponding
    to each label in the dataset.

    Attributes:
        data (Dataset): The Hugging Face dataset containing the data.
        transform (Optional[Callable]): An optional transform to be applied on the images.
        embeddings (torch.Tensor): A tensor containing precomputed embeddings for each label in the dataset.

    Args:
        hf_dataset (Dataset): The Hugging Face dataset to be used.
        embeddings (torch.Tensor): Precomputed embeddings corresponding to the labels in the dataset.
        transform (Optional[Callable], optional): A function/transform that takes in an image and returns a transformed version. Default is None.
    """

    def __init__(
        self,
        hf_dataset: Dataset,
        embeddings: torch.Tensor,
        transform: Optional[Callable] = None,
    ):
        assert (
            len(hf_dataset.features["label"].names) == embeddings.shape[0]
        ), "The number of labels in the dataset must match the number of rows in the embeddings tensor."

        self.data = hf_dataset
        self.transform = transform
        self.embeddings = embeddings

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieves the image and its corresponding text embedding at the specified index in the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding text embedding.
        """
        sample = self.data[idx]
        img = sample["img"]
        label = sample["label"]

        if self.transform is not None:
            img = self.transform(img)

        text_embedding = self.embeddings[label]

        return img, text_embedding.clone()


def make(config: Any) -> Tuple[Any, torch.Tensor, List[str]]:
    """
    Sets up and returns the components necessary for training an Imagen model on the CIFAR-10 dataset.

    This function loads the CIFAR-10 dataset, generates text embeddings for the labels, initializes the Unet and Imagen models,
    and sets up the training and validation datasets. It also configures the Imagen trainer with these datasets.

    Args:
        config (Any): A configuration object containing parameters for model initialization and training.
                      Expected to have attributes like dim, cond_dim, dim_mults, num_resnet_blocks, image_sizes,
                      timesteps, cond_drop_prob, dynamic_thresholding, lr, and batch_size.

    Returns:
        Tuple[Any, torch.Tensor, List[str]]: A tuple containing the initialized Imagen trainer, text embeddings,
                                             and a list of label names from the CIFAR-10 dataset.
    """
    unet = Unet(
        dim=config.dim,  # the "Z" layer dimension, i.e. the number of filters the outputs to the first layer
        cond_dim=config.cond_dim,
        dim_mults=config.dim_mults,  # the channel dimensions inside the model (multiplied by dim)
        num_resnet_blocks=config.num_resnet_blocks,
        layer_attns=(False,) + (True,) * (len(config.dim_mults) - 1),
        layer_cross_attns=(False,) + (True,) * (len(config.dim_mults) - 1),
    )

    imagen = Imagen(
        unets=unet,
        image_sizes=config.image_sizes,
        timesteps=config.timesteps,
        cond_drop_prob=config.cond_drop_prob,
        dynamic_thresholding=config.dynamic_thresholding,
    ).cuda()

    trainer = ImagenTrainer(imagen, lr=config.lr)

    train_dataset, valid_dataset = prepare_data()
    trainer.add_train_dataset(train_dataset, batch_size=config.batch_size)
    trainer.add_valid_dataset(valid_dataset, batch_size=config.batch_size)
    # train_dl, valid_dl = prepare_data()
    # trainer.add_train_dataloader(train_dl)
    # trainer.add_valid_dataloader(valid_dl)

    return trainer


def train(
    trainer,
    config,
    validate_every=None,
    save_every=None,
):
    """
    Trains a model using the provided trainer, configuration, and data.

    This function handles the training loop, including logging training and validation losses,
    generating and logging sample images at specified intervals, and saving the model periodically.

    Args:
        trainer (Any): The training object that handles the training steps.
        text_embeddings (torch.Tensor): Precomputed text embeddings corresponding to the labels.
        labels (List[str]): A list of label names corresponding to the text embeddings.
        config (Any): A configuration object containing training parameters and settings.
                      Expected to have attributes like steps, batch_size, and model_save_dir.
        sample_factor (Optional[float], optional): A factor to determine the interval for generating samples.
                                                   If provided, sample_every interval will be multiplied by this factor after each sampling. Default is None.
        validate_every (Optional[int], optional): The interval (in number of steps) at which to perform validation.
                                                  If None, validation is skipped. Default is None.
        save_every (Optional[int], optional): The interval (in number of steps) at which to save the model.
                                              If None, saving is skipped. Default is None.

    Raises:
        AssertionError: If the model_save_dir in config does not end with a '/'.

    Notes:
        - The function uses Weights & Biases (wandb) for logging.
        - The model is saved with the naming convention "{model_save_dir}{wandb.run.name}-{step}.ckpt".
        - Validation loss is computed as the average loss over 100 validation steps.
        - The function assumes that the trainer object has 'train_step', 'valid_step', 'sample', and 'save' methods.
    """
    assert config.model_save_dir[-1] == "/"

    def next_sample_step(current_step):
        if current_step < 10:
            return current_step + 1
        else:
            return int(current_step * 3)

    sample_step = 0

    for i in range(config.steps):
        loss = trainer.train_step(max_batch_size=config.batch_size)

        wandb.log({"train_loss": loss}, step=i)

        # TODO: add validation loss (can do every sampling step)
        if validate_every is not None and i % validate_every == 0:
            avg_loss = 0
            for _ in range(100):
                valid_loss = trainer.valid_step(
                    unet_number=1, max_batch_size=config.batch_size
                )
                avg_loss += valid_loss
            wandb.log({"valid loss": avg_loss}, step=i)

        if i == sample_step:
            # Sample a batch from the DataLoader
            sample_batch = next(iter(trainer.train_dataloader))
            (
                sample_images,
                _,
            ) = sample_batch  # Assuming each batch is a tuple (images, embeddings)

            # Generate samples using the model
            generated_images = trainer.sample(sample_images)

            # Log the generated images
            samples = [wandb.Image(img) for img in generated_images]
            wandb.log({"samples": samples}, step=i)

            sample_step = next_sample_step(i)

        if save_every is not None and i != 0 and i % save_every == 0:
            trainer.save(f"{config.model_save_dir}{wandb.run.name}-{i}.ckpt")

    # final save at the end if we did not already save this round
    if save_every is not None and i % save_every != 0:
        trainer.save(f"{config.model_save_dir}{wandb.run.name}-{i}.ckpt")


def build(hyperparams: dict) -> Any:
    """
    Initializes and starts a training session with given hyperparameters using Weights & Biases (wandb).

    This function sets up a training environment in wandb, creates a trainer, and starts the training process.
    It uses the hyperparameters provided to configure the training session and logs the training process
    and results in the specified wandb project.

    Args:
        hyperparams (dict): A dictionary containing hyperparameters for the training session.
                            These parameters are used to configure the model, trainer, and training process.

    Returns:
        Any: The trained model's trainer object after completing the training process.

    Notes:
        - The function assumes the existence of global variables: WANDB_PROJECT_NAME, EXPERIMENT_NAME, and TAGS,
          which are used to initialize the wandb training session.
        - The function internally calls `make` to create a trainer and `train` to start the training process.
        - The `sample_factor`, `validate_every`, and `save_every` parameters for the `train` function are set within this function.
        - The wandb session logs the training process, including parameters and metrics, in the project defined by WANDB_PROJECT_NAME.
    """
    with wandb.init(
        project=WANDB_PROJECT_NAME, name=EXPERIMENT_NAME, tags=TAGS, config=hyperparams
    ):
        config = wandb.config  # hyperparams

        trainer = make(config)

        train(
            trainer,
            config,
            validate_every=None,
            save_every=10_000,
        )

        return trainer


trainer = build(HYPERPARAMS)
