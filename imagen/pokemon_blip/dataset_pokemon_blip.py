import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import T5Tokenizer, T5EncoderModel
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import pickle
from torchvision import transforms


TEXT_ENCODER = "google/t5-v1_1-base"
HF_DATASET = "lambdalabs/pokemon-blip-captions"

DATASET_FNAME = "pokemon_blip_dataset.hf"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(TEXT_ENCODER, model_max_length=256)
model = T5EncoderModel.from_pretrained(TEXT_ENCODER)
model.eval()
model = model.to(device)


class HFDataset(Dataset):
    """Interface between the HuggingFace dataset and PyTorch."""

    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = sample["image"]
        caption_embedding = sample["embedding"]

        if self.transform is not None:
            img = self.transform(img)

        return img, caption_embedding.clone()


def append_pokemon_style(batch):
    """
    Appends 'in the style of Pokemon' to each text element in the batch.
    """
    # Append the string to each element in the 'text' field of the batch
    batch["text"] = [text + " in the style of Pokemon" for text in batch["text"]]
    return batch


def generate_batch_embeddings(batch, max_length: int = 64):
    """Generate embeddings for a batch of captions."""
    # Tokenize the captions in the current batch
    encoded = tokenizer.batch_encode_plus(
        batch["text"],  # image, text pairs
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    # Move encoded inputs to the device
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    # Generate embeddings
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        encoded_text = output.last_hidden_state.detach()

    # Move embeddings back to CPU for further processing/storage
    encoded_text = encoded_text.to("cpu")

    attn_mask = attention_mask.to("cpu").bool()
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, "... -> ... 1"), 0.0)
    return {"embedding": torch.Tensor(encoded_text)}


def prepare_dataset():
    """Prepare the Pokemon BLIP dataset for training."""
    if os.path.exists(DATASET_FNAME):
        pkmn_ds = load_from_disk(DATASET_FNAME)

    else:
        pkmn_ds = load_dataset(HF_DATASET)

        pkmn_ds = pkmn_ds.map(
            append_pokemon_style,
            batched=True,
            batch_size=16,
        )

        pkmn_ds = pkmn_ds.map(
            lambda batch: generate_batch_embeddings(batch, max_length=64),
            batched=True,
            batch_size=16,
        )

        pkmn_ds.save_to_disk(DATASET_FNAME)

    # Create dataset instances
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),  # ImageNet mean and std dev
        ]
    )

    return HFDataset(
        hf_dataset=pkmn_ds["train"].with_format("torch"), transform=transform
    )
