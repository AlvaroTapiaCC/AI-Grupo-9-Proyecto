import shutil
import torch
from tqdm import tqdm

from ..paths import (
    TRAIN_ANNOTATIONS,
    TEST_ANNOTATIONS,
    VAL_ANNOTATIONS,
    CATEGORIES_PATH,
    TENSORS_PATH,
    TRAIN_TENS,
    TEST_TENS,
    VAL_TENS,
)
from ..data.label_encoder import LabelEncoder
from ..utils.io import load_json
from ..data.data_utils import build_category_mapping, build_image_mapping
from ..data.dataset_loader import RetailDataset
from ..data.transforms import get_val_transforms


TENSORS_PATH.mkdir(parents=True, exist_ok=True)


def clear_tensors():
    if TENSORS_PATH.exists():
        shutil.rmtree(TENSORS_PATH)
    TENSORS_PATH.mkdir(parents=True, exist_ok=True)


def process_split(split_name, ann_path, cat_map, label_encoder: LabelEncoder, save_path):
    print(f"\n[INFO] Processing {split_name}...")

    dataset = RetailDataset(
        ann_path,
        split=split_name,
        transform=get_val_transforms(),
        label_encoder_path=None,
        preload=False
    )

    all_images = []
    raw_labels = []

    for i in tqdm(range(len(dataset))):
        x, _ = dataset[i]

        ann = dataset.samples[i]
        _, _, supercat_id = ann

        all_images.append(x)
        raw_labels.append(supercat_id)

    if len(all_images) == 0:
        print(f"[WARNING] No tensors generated for {split_name}")
        return

    images = torch.stack(all_images)

    labels = torch.tensor(
        label_encoder.transform(raw_labels),
        dtype=torch.long
    )

    torch.save(
        {
            "images": images,
            "labels": labels,
        },
        save_path,
    )

    print(f"[OK] Saved {split_name}: {images.shape}")


def build_all_tensors():
    print("[INFO] Clearing old tensors...")
    clear_tensors()

    cats_json = load_json(CATEGORIES_PATH)
    cat_map = build_category_mapping(cats_json)

    all_labels = []

    for ann_path in [TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, TEST_ANNOTATIONS]:
        data = load_json(ann_path)
        for ann in data["annotations"]:
            supercat_id = cat_map.get(ann["category_id"])
            if supercat_id is not None:
                all_labels.append(supercat_id)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    label_encoder.save(TENSORS_PATH / "label_encoder.json")

    process_split("train", TRAIN_ANNOTATIONS, cat_map, label_encoder, TRAIN_TENS)
    process_split("val", VAL_ANNOTATIONS, cat_map, label_encoder, VAL_TENS)
    process_split("test", TEST_ANNOTATIONS, cat_map, label_encoder, TEST_TENS)

    print("\n[DONE]")