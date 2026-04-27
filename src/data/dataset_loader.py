import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image

from ..paths import IMAGES_PATH
from .label_encoder import LabelEncoder
from ..utils.io import load_json


def load_embeddings(path):
    data = torch.load(path)
    return TensorDataset(data["embeddings"], data["labels"])



class RetailDataset(Dataset):
    def __init__(self, annotations_path, split, transform=None, label_encoder_path=None):
        self.data = load_json(annotations_path)
        self.split = split
        self.transform = transform

        self.image_map = {
            img["id"]: img["file_name"]
            for img in self.data["images"]
        }

        self.cat_map = {
            c["id"]: c["supercat_id"]
            for c in self.data["categories"]
        }

        self.label_encoder = (
            LabelEncoder.load(label_encoder_path)
            if label_encoder_path is not None
            else None
        )

        self.samples = []

        for ann in self.data["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]

            file_name = self.image_map.get(image_id)
            if file_name is None:
                continue

            supercat_id = self.cat_map.get(category_id)
            if supercat_id is None:
                continue

            if self.label_encoder is not None:
                label = self.label_encoder.id2idx.get(supercat_id, None)
                if label is None:
                    continue
            else:
                label = supercat_id

            bbox = ann.get("bbox", None)
            if bbox is None:
                continue

            self.samples.append((file_name, bbox, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, bbox, label = self.samples[idx]

        image_path = IMAGES_PATH / self.split / file_name
        image = Image.open(image_path).convert("RGB")

        x, y, w, h = bbox

        width, height = image.size
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int = 2
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader