import json
import shutil
import torch
from pathlib import Path
from PIL import Image
import clip

from ..config import batch_size, device
from ..paths import (
    IMAGES_PATH,
    TRAIN_ANNOTATIONS,
    TEST_ANNOTATIONS,
    VAL_ANNOTATIONS,
    CATEGORIES_PATH,
    EMBEDDINGS_PATH,
)
from ..data.label_encoder import LabelEncoder
from ..utils.io import load_json
from ..data.data_utils import build_category_mapping, build_image_mapping


# ensure base dir exists
EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


def clear_embeddings():
    if EMBEDDINGS_PATH.exists():
        shutil.rmtree(EMBEDDINGS_PATH)
    EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)


def process_split(split_name, ann_path, cat_map, label_encoder: LabelEncoder):
    print(f"\n[INFO] Processing {split_name}...")

    ann_data = load_json(ann_path)
    image_map = build_image_mapping(ann_data["images"])

    embeddings = []
    raw_labels = []

    batch_images = []
    batch_labels = []

    current_image_id = None
    current_image = None

    for ann in ann_data["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        category_id = ann["category_id"]

        file_name = image_map.get(image_id)
        if file_name is None:
            continue

        image_path = IMAGES_PATH / split_name / file_name
        if not image_path.exists():
            continue

        if image_id != current_image_id:
            current_image = Image.open(image_path).convert("RGB")
            current_image_id = image_id

        x, y, w, h = bbox
        crop = current_image.crop((x, y, x + w, y + h))
        crop = preprocess(crop)

        supercat_id = cat_map.get(category_id)
        if supercat_id is None:
            continue

        batch_images.append(crop)
        batch_labels.append(supercat_id)

        if len(batch_images) == batch_size:
            batch = torch.stack(batch_images).to(device)

            with torch.no_grad():
                emb = model.encode_image(batch)

            embeddings.append(emb.cpu())
            raw_labels.extend(batch_labels)

            batch_images = []
            batch_labels = []

    if batch_images:
        batch = torch.stack(batch_images).to(device)

        with torch.no_grad():
            emb = model.encode_image(batch).float()

        embeddings.append(emb.cpu())
        raw_labels.extend(batch_labels)

    if len(embeddings) == 0:
        print(f"[WARNING] No embeddings generated for {split_name}")
        return

    embeddings = torch.cat(embeddings, dim=0)

    labels = torch.tensor(
        label_encoder.transform(raw_labels),
        dtype=torch.long
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        EMBEDDINGS_PATH / f"{split_name}.pt",
    )

    print(f"[OK] Saved {split_name}: {embeddings.shape}")


def build_embeddings():
    print("[INFO] Clearing old embeddings...")
    clear_embeddings()

    cats_json = load_json(CATEGORIES_PATH)
    cat_map = build_category_mapping(cats_json)

    # build global label encoder from all splits
    all_labels = []

    for ann_path in [TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, TEST_ANNOTATIONS]:
        data = load_json(ann_path)
        for ann in data["annotations"]:
            supercat_id = cat_map.get(ann["category_id"])
            if supercat_id is not None:
                all_labels.append(supercat_id)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    label_encoder.save(EMBEDDINGS_PATH / "label_encoder.json")

    process_split("train", TRAIN_ANNOTATIONS, cat_map, label_encoder)
    process_split("val", VAL_ANNOTATIONS, cat_map, label_encoder)
    process_split("test", TEST_ANNOTATIONS, cat_map, label_encoder)

    print("\n[DONE]")