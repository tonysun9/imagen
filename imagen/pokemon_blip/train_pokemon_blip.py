import numpy as np
import random
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torch.utils.data import Dataset
import os
import wandb

from typing import Any, List, Tuple

from dataset_pkmn import prepare_dataset, generate_batch_embeddings

WANDB_PROJECT_NAME = "imagen-pkmn"
EXPERIMENT_NAME = "pkmn-test-1"
TAGS = ["pkmn", "imagen"]

MODEL_SAVE_DIR = "pkmn_checkpoints/"

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
    "image_sizes": 32,  # The size (height and width) of the images that the model will process and generate.
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


CAPTIONS = {
    "text": [
        "a drawing of a green pokemon with red eyes in the style of Pokemon.",
        "a cartoon of a fire breathing dragon in the style of Pokemon.",
        "a cartoon pikachu with a big smile on his face in the style of Pokemon.",
        "A bright orange creature with a flickering flame at the end of its tail and intense, fiery eyes in the style of Pokemon.",
        "A small, quadrupedal creature with a bulb on its back, surrounded by lush green foliage in the style of Pokemon.",
        "A small, yellow rodent-like creature with pointed ears, red cheeks, and a lightning bolt-shaped tail in the style of Pokemon.",
        "With sharp, burning eyes that seem to flicker with an intensity that could melt steel, this towering creature possesses an unmistakable aura of power. Its body is enveloped in fiery red-orange scales, which appear impenetrable and shimmer with a serpentine grace. Gigantic wings spread out majestically from its back, their span accentuated by vibrant, flame-like patterns that ignite the imagination in the style of Pokemon.",
        "This creature boasts a small, reptilian body covered in a thick, turquoise skin. It possesses large, round eyes that exude an innocent charm, accentuated by its dark, kind eyes staring out from beneath a pair of broad, leaf-like ears. From a peculiar bulb on its back, two vibrant green leaves emerge, creating an enchanting sight that hints at its nature-based abilities in the style of Pokemon.",
        "With its electrifying presence, this majestic creature stands tall and proud, boasting vibrant plumage in shades of brilliant yellow and electric blue. Its wings span wide, showcasing a striking pattern resembling bolts of lightning, which crackle and shimmer with electrifying energy. A fierce gaze emanates from its piercing red eyes, commanding both authority and an aura of wild power.",
    ]
}


# pokemon_list = [
#     "Bulbasaur",
#     "Charizard",
#     "Pikachu",
#     "Charizard",
#     "Bulbasaur",
#     "Pikachu",
#     "Charizard (ChatGPT)",
#     "Bulbasaur (ChatGPT)",
#     "Zapdos (ChatGPT)",
# ]


CAPTION_EMBEDDINGS = generate_batch_embeddings(batch=CAPTIONS)["embedding"]

if not os.path.exists(MODEL_SAVE_DIR):
    os.mkdir(MODEL_SAVE_DIR)

wandb.login(key=os.environ["WANDB_API_KEY"])


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

    # train_dataset, valid_dataset = prepare_data()
    train_dataset = prepare_dataset()
    trainer.add_train_dataset(train_dataset, batch_size=config.batch_size)
    # trainer.add_valid_dataset(valid_dataset, batch_size=config.batch_size)

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

    sample_every = 1000

    for i in range(config.steps):
        loss = trainer.train_step(max_batch_size=config.batch_size)

        wandb.log({"train_loss": loss}, step=i)

        # TODO: add validation loss (can do every sampling step)
        # if validate_every is not None and i % validate_every == 0:
        #     avg_loss = 0
        #     for _ in range(100):
        #         valid_loss = trainer.valid_step(
        #             unet_number=1, max_batch_size=config.batch_size
        #         )
        #         avg_loss += valid_loss
        #     wandb.log({"valid loss": avg_loss}, step=i)

        if i % sample_every == 0 or i in {10, 100, 300}:
            # Sample a batch from the DataLoader
            images = trainer.sample(
                text_embeds=CAPTION_EMBEDDINGS,
                batch_size=config.batch_size,
                return_pil_images=True,
            )
            samples = []
            for j, img in enumerate(images):
                samples.append(wandb.Image(img, caption=CAPTIONS["caption"][j]))
            wandb.log({"samples": samples}, step=i)

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
