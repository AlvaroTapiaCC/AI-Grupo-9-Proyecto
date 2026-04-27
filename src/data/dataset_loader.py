import torch
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ..paths import IMAGES_PATH, CATEGORIES_PATH
from .label_encoder import LabelEncoder
from ..utils.io import load_json
from .data_utils import build_category_mapping, build_image_mapping


def load_embeddings(path):
    data = torch.load(path)
    return TensorDataset(data["embeddings"], data["labels"])


class RetailDataset(Dataset):
    def __init__(self, annotations_path, split, transform=None, label_encoder_path=None, preload=False):
        self.split = split
        self.transform = transform
        self.preload = preload

        ann_json = load_json(annotations_path)
        cat_json = load_json(CATEGORIES_PATH)

        self.image_map = build_image_mapping(ann_json["images"])
        self.cat_map = build_category_mapping(cat_json)

        self.label_encoder = (
            LabelEncoder.load(label_encoder_path)
            if label_encoder_path is not None
            else None
        )

        self.samples = []

        for ann in ann_json["annotations"]:
            image_id = ann["image_id"]
            bbox = ann["bbox"]
            category_id = ann["category_id"]

            file_name = self.image_map.get(image_id)
            if file_name is None:
                continue

            supercat_id = self.cat_map.get(category_id)
            if supercat_id is None:
                continue

            if self.label_encoder is not None:
                label = self.label_encoder.id2idx.get(supercat_id)
                if label is None:
                    continue
            else:
                label = supercat_id

            self.samples.append((file_name, bbox, label))

        self.data = []

        if self.preload:
            print(f"[INFO] Preloading {split} dataset into RAM...")

            for file_name, bbox, label in self.samples:
                image_path = str(IMAGES_PATH / split / file_name)

                image = io.read_image(image_path)  # [C, H, W], uint8

                x, y, w, h = bbox
                _, H, W = image.shape

                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(W, int(x + w))
                y2 = min(H, int(y + h))

                image = image[:, y1:y2, x1:x2]
                image = image.float() / 255.0

                if self.transform:
                    image = self.transform(image)

                self.data.append(
                    (image, torch.tensor(label, dtype=torch.long))
                )

            print(f"[INFO] {split} preloaded: {len(self.data)} samples")

    def __len__(self):
        return len(self.data if self.preload else self.samples)

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]

        file_name, bbox, label = self.samples[idx]

        image_path = str(IMAGES_PATH / self.split / file_name)

        image = io.read_image(image_path)

        x, y, w, h = bbox
        _, H, W = image.shape

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W, int(x + w))
        y2 = min(H, int(y + h))

        image = image[:, y1:y2, x1:x2]
        image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(
    train_dataset,
    test_dataset,
    val_dataset,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    return train_loader, test_loader, val_loader