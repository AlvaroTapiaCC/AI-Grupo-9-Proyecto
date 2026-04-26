import json
from PIL import Image
from torch.utils.data import Dataset
import torch

from ..paths import IMAGES_PATH
from .label_encoder import LabelEncoder


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


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

            label = (
                self.label_encoder.id2idx[supercat_id]
                if self.label_encoder is not None
                else supercat_id
            )

            self.samples.append((file_name, ann["bbox"], label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, bbox, label = self.samples[idx]

        image_path = IMAGES_PATH / self.split / file_name
        image = Image.open(image_path).convert("RGB")

        x, y, w, h = bbox
        image = image.crop((x, y, x + w, y + h))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)