import numpy as np
import random
from datasets import load_dataset
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import T5Tokenizer, T5EncoderModel
from einops import rearrange
import os
import wandb

from typing import Optional, Callable, Any, List, Tuple

WANDB_PROJECT_NAME = "imagen"
EXPERIMENT_NAME = "coco-test-1"
TAGS = ["coco", "imagen", "original"]

MODEL_SAVE_DIR = "coco_checkpoints/"

TEXT_ENCODER = "google/t5-v1_1-base"

SEED = 0

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


def get_text_embeddings(
    name: str, labels: list[str], max_length: int = 256
) -> torch.Tensor:
    """
    Generates or loads text embeddings for a given list of labels using the T5 model.

    This function checks if the embeddings are already saved in a file with the given name.
    If not, it loads the T5 model and tokenizer, processes the labels to generate embeddings,
    and saves these embeddings to the specified file.

    Args:
    name (str): The file name where the embeddings are saved or will be saved.
    labels (List[str]): A list of labels for which embeddings are to be generated.
    max_length (int, optional): The maximum sequence length for the T5 tokenizer. Default is 256.

    Returns:
    torch.Tensor: A tensor containing the generated embeddings.
    """

    if os.path.isfile(name):
        return torch.load(name)

    tokenizer = T5Tokenizer.from_pretrained(TEXT_ENCODER, model_max_length=max_length)

    model = T5EncoderModel.from_pretrained(TEXT_ENCODER)
    model.eval()

    def photo_prefix(noun: str) -> str:
        """Adds a prefix to a noun to create a phrase."""
        if noun[0] in ["a", "e", "i", "o", "u"]:
            return "a photo of an " + noun
        return "a photo of a " + noun

    texts = [photo_prefix(x) for x in labels]

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    )

    with torch.no_grad():
        output = model(
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask
        )
        encoded_text = output.last_hidden_state.detach()

    attn_mask = encoded.attention_mask.bool()

    # Mask the encoded text where attention mask is false
    # encoded_text = encoded_text.masked_fill(
    #     ~torch.einsum("... -> ... 1", attn_mask), 0.0
    # )
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, "... -> ... 1"), 0.0)

    torch.save(encoded_text, name)

    return encoded_text


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
    cfar = load_dataset("cifar10")
    labels = cfar["train"].features["label"].names
    text_embeddings = get_text_embeddings("cifar10-embeddings.pkl", labels)

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

    transform = transforms.Compose(
        [
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet mean and std dev
        ]
    )

    ds = HFDataset(cfar["train"], text_embeddings, transform=transform)
    tst_ds = HFDataset(cfar["test"], text_embeddings, transform=transform)

    trainer.add_train_dataset(ds, batch_size=config.batch_size)
    trainer.add_valid_dataset(tst_ds, batch_size=config.batch_size)

    return trainer, text_embeddings, labels


def train(
    trainer,
    text_embeddings,
    labels,
    config,
    sample_factor=None,
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

    sample_every = 10

    for i in range(config.steps):
        loss = trainer.train_step(max_batch_size=config.batch_size)

        wandb.log({"train_loss": loss}, step=i)

        if validate_every is not None and i % validate_every == 0:
            avg_loss = 0
            for _ in range(100):
                valid_loss = trainer.valid_step(
                    unet_number=1, max_batch_size=config.batch_size
                )
                avg_loss += valid_loss
            wandb.log({"valid loss": avg_loss}, step=i)

        if sample_factor is not None and i % sample_every == 0:
            images = trainer.sample(
                text_embeds=text_embeddings,
                batch_size=config.batch_size,
                return_pil_images=True,
            )
            samples = []
            for j, img in enumerate(images):
                samples.append(wandb.Image(img, caption=labels[j]))
            wandb.log({"samples": samples}, step=i)
            sample_every = int(sample_every * sample_factor)

        if save_every is not None and i != 0 and i % save_every == 0:
            trainer.save(f"{config.model_save_dir}{wandb.run.name}-{i}.ckpt")

    # final save at the end if we did not already save this round
    if save_every is not None and i % save_every != 0:
        trainer.save(f"{config.model_save_dir}{wandb.run.name}-{i}.ckpt")


hyperparams = {
    "steps": 200_000,  # Total number of training steps (iterations) to run.
    "dim": 128,  # Base dimensionality of the model, often related to the complexity and capacity of the model.
    "cond_dim": 128,  # Dimensionality of the conditional input, e.g., for conditional GANs or conditional image generation.
    "dim_mults": (
        1,
        2,
        4,
    ),  # Multipliers for the dimensions at different stages in the network, controlling the depth and width of the model.
    "image_sizes": 32,  # The size (height and width) of the images that the model will process and generate.
    "timesteps": 250,
    "cond_drop_prob": 0.1,  # Dropout probability for the conditional inputs, used for regularization.
    "batch_size": 64,
    "lr": 1e-4,
    "num_resnet_blocks": 3,
    "model_save_dir": MODEL_SAVE_DIR,
    "dynamic_thresholding": True,
}

config = wandb.config


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

        trainer, embeddings, labels = make(config)

        train(
            trainer,
            embeddings,
            labels,
            config,
            sample_factor=1.3,
            validate_every=None,
            save_every=10_000,
        )

        return trainer


trainer = build(hyperparams)
