from tqdm import tqdm_notebook as tqdm
import numpy as np
import random
from datasets import load_dataset
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from einops import rearrange
import os
import wandb

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# We will be saving checkpoints to our google drive so we can download them
# later
model_save_dir = "cifar_checkpoints/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

wandb.login(key=os.environ["WANDB_API_KEY"])


def get_text_embeddings(name, labels, max_length=256):
    if os.path.isfile(name):
        return torch.load(name)

    model_name = "google/t5-v1_1-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_length)

    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    def photo_prefix(noun):
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

    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, "... -> ... 1"), 0.0)

    torch.save(encoded_text, name)

    return encoded_text


class HFDataset(Dataset):
    def __init__(self, hf_dataset, embeddings, transform=None):
        assert len(hf_dataset.features["label"].names) == embeddings.shape[0]

        self.data = hf_dataset
        self.transform = transform
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = sample["img"]
        label = sample["label"]

        if self.transform is not None:
            img = self.transform(img)

        text_embedding = self.embeddings[label]

        return img, text_embedding.clone()


def make(config):
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

    ds = HFDataset(
        cfar["train"],
        text_embeddings,
        transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
    )

    tst_ds = HFDataset(
        cfar["test"],
        text_embeddings,
        transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
    )

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
    "steps": 200_000,
    "dim": 128,
    "cond_dim": 128,
    "dim_mults": (1, 2, 4),
    "image_sizes": 32,
    "timesteps": 250,
    "cond_drop_prob": 0.1,
    "batch_size": 64,
    "lr": 1e-4,
    "num_resnet_blocks": 3,
    "model_save_dir": model_save_dir,
    "dynamic_thresholding": True,
}

config = wandb.config


def build(hyperparams):
    with wandb.init(project="cifar10-imagen", config=hyperparams):
        config = wandb.config

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
