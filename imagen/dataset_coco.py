import os
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5EncoderModel
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


TEXT_ENCODER = "google/t5-v1_1-base"
PATH_TO_IMAGE_FOLDER = "COCO2017"

BATCH_SIZE = 64


def create_full_path(image):
    """Create full path to image using `base_path` to COCO2017 folder."""
    image["image_path"] = os.path.join(PATH_TO_IMAGE_FOLDER, image["file_name"])
    return image


def preprocess_coco(dataset):
    """Flatten the dataset so each image-caption pair is a separate data point."""
    image_caption_pairs = []
    for item in dataset:
        for caption in item["captions"]:
            image_caption_pairs.append(
                {
                    "image_path": os.path.join(PATH_TO_IMAGE_FOLDER, item["file_name"]),
                    "caption": caption,
                }
            )
    return image_caption_pairs


def generate_text_embeddings(
    name: str, captions: list[str], max_length: int = 256
) -> torch.Tensor:
    """
    Generates or loads text embeddings for a given list of captions using the T5 model.

    This function checks if the embeddings are already saved in a file with the given name.
    If not, it loads the T5 model and tokenizer, processes the captions to generate embeddings,
    and saves these embeddings to the specified file.

    Args:
        name (str): The file name where the embeddings are saved or will be saved.
        captions (List[str]): A list of captions for which embeddings are to be generated.
        max_length (int, optional): The maximum sequence length for the T5 tokenizer. Default is 256.

    Returns:
        torch.Tensor: A tensor containing the generated embeddings.
    """

    if os.path.isfile(name):
        return torch.load(name)

    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(TEXT_ENCODER, model_max_length=max_length)
    model = T5EncoderModel.from_pretrained(TEXT_ENCODER)
    model.eval()

    # Tokenize the captions
    encoded = tokenizer.batch_encode_plus(  # a method from the Hugging Face transformers library, used to convert a batch of text strings (in this case, captions) into a format that can be processed by the T5 model.
        captions,
        return_tensors="pt",  # output should be PyTorch tensors
        padding="longest",  # pad the sequences to the longest sequence in the batch
        max_length=max_length,
        truncation=True,
    )

    # Generate embeddings
    with torch.no_grad():  # isable gradient calculations, reducing memory usage and speeding up the process. It's typically used during inference.
        output = model(
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask
        )
        encoded_text = output.last_hidden_state.detach()

    attn_mask = encoded.attention_mask.bool()

    # Mask the encoded text where attention mask is false
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, "... -> ... 1"), 0.0)

    # Save the embeddings
    torch.save(encoded_text, name)

    return encoded_text


class CustomCocoDataset(Dataset):
    def __init__(self, image_caption_pairs, transform=None):
        self.image_caption_pairs = image_caption_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        pair = self.image_caption_pairs[idx]
        image = Image.open(pair["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        embedding = pair["embedding"]
        return image, embedding


def prepare_data():
    # Load and preprocess the COCO dataset
    coco_dataset = load_dataset("phiyodr/coco2017")
    coco_dataset = coco_dataset.map(create_full_path)

    train_image_caption_pairs = preprocess_coco(coco_dataset["train"])
    valid_image_caption_pairs = preprocess_coco(
        coco_dataset["validation"]
    )  # Assuming validation data is available

    # Extract captions and generate embeddings
    all_captions = [pair["caption"] for pair in train_image_caption_pairs]
    all_embeddings = generate_text_embeddings("embeddings_file_name.pkl", all_captions)

    # Add embeddings to the pairs
    for pair, embedding in zip(train_image_caption_pairs, all_embeddings):
        pair["embedding"] = embedding

    # Create dataset instances
    transform = transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet mean and std dev
        ]
    )

    train_dataset = CustomCocoDataset(train_image_caption_pairs, transform=transform)
    valid_dataset = CustomCocoDataset(valid_image_caption_pairs, transform=transform)

    return train_dataset, valid_dataset

    # Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    # return train_loader, valid_loader
