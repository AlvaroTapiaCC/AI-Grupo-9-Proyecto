import shutil
import torch
import torch.nn as nn
import torchvision.io as io
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .. import config
from ..paths import (
    IMAGES_PATH,
    TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, TEST_ANNOTATIONS,
    DINO_EMB_PATH, DINO_LABEL_ENCODER,
    DINO_TRAIN_EMB, DINO_VAL_EMB, DINO_TEST_EMB,
    DETECTOR_FEAT_PATH,
    DETECTOR_TRAIN_FEAT, DETECTOR_VAL_FEAT, DETECTOR_TEST_FEAT,
    CATEGORIES_PATH,
)
from ..data.label_encoder import LabelEncoder
from ..utils.io import load_json
from ..data.data_utils import build_category_mapping, build_image_mapping


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_CROP_SPLIT_PATHS = {
    "train": DINO_TRAIN_EMB,
    "val":   DINO_VAL_EMB,
    "test":  DINO_TEST_EMB,
}

_DET_SPLIT_PATHS = {
    "train": DETECTOR_TRAIN_FEAT,
    "val":   DETECTOR_VAL_FEAT,
    "test":  DETECTOR_TEST_FEAT,
}


class DINOv2Encoder(nn.Module):
    MODEL_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_name: str = "dinov2_vitb14", freeze: bool = True):
        super().__init__()
        self.backbone    = torch.hub.load("facebookresearch/dinov2", model_name)
        self.feature_dim = self.MODEL_DIMS[model_name]
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns CLS token: (B, feature_dim)."""
        return self.backbone.forward_features(x)["x_norm_clstoken"]

    def forward_detector(self, x: torch.Tensor):
        """Returns (cls_token, patch_tokens): (B, D) and (B, N_patches, D)."""
        feats = self.backbone.forward_features(x)
        return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]


def _preprocess(img_tensor: torch.Tensor, bbox=None) -> torch.Tensor:
    """
    img_tensor: (C, H, W) uint8
    bbox: (x, y, w, h) to crop, or None for full image.
    Returns (C, 224, 224) float32 ImageNet-normalized.
    """
    if bbox is not None:
        x, y, w, h = bbox
        _, H, W = img_tensor.shape
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        img_tensor = img_tensor[:, y1:y2, x1:x2]
    img = TF.resize(img_tensor, [224, 224], antialias=True)
    return (img.float() / 255.0 - _MEAN) / _STD


# ── Crop embeddings (DINOv2 classifier) ──────────────────────────────────────

def _process_crop_split(split_name, ann_path, cat_map, label_encoder, encoder):
    print(f"\n[INFO] Processing {split_name}...")
    ann_data  = load_json(ann_path)
    image_map = build_image_mapping(ann_data["images"])

    all_embs, raw_labels = [], []
    batch_crops, batch_labels = [], []
    current_image_id, current_image = None, None

    for ann in tqdm(ann_data["annotations"], desc=split_name, ncols=100):
        image_id    = ann["image_id"]
        supercat_id = cat_map.get(ann["category_id"])
        if supercat_id is None:
            continue
        file_name = image_map.get(image_id)
        if file_name is None:
            continue
        image_path = IMAGES_PATH / split_name / file_name
        if not image_path.exists():
            continue

        if image_id != current_image_id:
            current_image    = io.read_image(str(image_path))
            current_image_id = image_id

        batch_crops.append(_preprocess(current_image, ann["bbox"]))
        batch_labels.append(supercat_id)

        if len(batch_crops) == config.batch_size:
            batch = torch.stack(batch_crops).to(config.device)
            with torch.no_grad():
                all_embs.append(encoder(batch).cpu())
            raw_labels.extend(batch_labels)
            batch_crops, batch_labels = [], []

    if batch_crops:
        batch = torch.stack(batch_crops).to(config.device)
        with torch.no_grad():
            all_embs.append(encoder(batch).cpu())
        raw_labels.extend(batch_labels)

    if not all_embs:
        print(f"[WARNING] No embeddings for {split_name}")
        return

    embeddings = torch.cat(all_embs, dim=0)
    labels     = torch.tensor(label_encoder.transform(raw_labels), dtype=torch.long)
    torch.save({"embeddings": embeddings, "labels": labels}, _CROP_SPLIT_PATHS[split_name])
    print(f"[OK] {split_name}: {embeddings.shape}")


def build_crop_embeddings():
    """Precompute DINOv2 CLS token embeddings for GT bbox crops (classifier)."""
    print("[INFO] Clearing old DINOv2 crop embeddings...")
    if DINO_EMB_PATH.exists():
        shutil.rmtree(DINO_EMB_PATH)
    DINO_EMB_PATH.mkdir(parents=True, exist_ok=True)

    cats_json = load_json(CATEGORIES_PATH)
    cat_map   = build_category_mapping(cats_json)

    all_labels = []
    for ann_path in [TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, TEST_ANNOTATIONS]:
        for ann in load_json(ann_path)["annotations"]:
            sid = cat_map.get(ann["category_id"])
            if sid is not None:
                all_labels.append(sid)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    label_encoder.save(DINO_LABEL_ENCODER)

    encoder = DINOv2Encoder(model_name=config.dinov2_model, freeze=True).to(config.device)
    encoder.eval()

    _process_crop_split("train", TRAIN_ANNOTATIONS, cat_map, label_encoder, encoder)
    _process_crop_split("val",   VAL_ANNOTATIONS,   cat_map, label_encoder, encoder)
    _process_crop_split("test",  TEST_ANNOTATIONS,  cat_map, label_encoder, encoder)

    print("\n[DONE]")


# ── Full-image CLS features (new count+box detector) ─────────────────────────

def _normalize_box(box, img_w, img_h):
    """[x, y, w, h] COCO → [x1, y1, x2, y2] normalized to [0, 1]."""
    x, y, w, h = box
    return [
        max(0.0, x / img_w),
        max(0.0, y / img_h),
        min(1.0, (x + w) / img_w),
        min(1.0, (y + h) / img_h),
    ]


def _build_image_index(ann_path):
    """Returns {image_id: {file_name, width, height, boxes: [[x,y,w,h], ...]}}."""
    data      = load_json(ann_path)
    image_map = {img["id"]: img for img in data["images"]}
    index     = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in index:
            img = image_map[iid]
            index[iid] = {
                "file_name": img["file_name"],
                "width":     img["width"],
                "height":    img["height"],
                "boxes":     [],
            }
        index[iid]["boxes"].append(ann["bbox"])
    return index


def _process_detector_split(split_name, ann_path, encoder, save_path):
    print(f"\n[INFO] Processing {split_name}...")
    image_index = _build_image_index(ann_path)
    MAX_DET     = config.max_detections

    all_cls_tokens, all_patch_tokens, all_counts, all_boxes = [], [], [], []
    batch_imgs, batch_meta = [], []

    def _flush():
        batch = torch.stack(batch_imgs).to(config.device)
        with torch.no_grad():
            cls_tokens, patch_tokens = encoder.forward_detector(batch)
        cls_tokens   = cls_tokens.cpu()
        patch_tokens = patch_tokens.cpu()
        for i, (count, boxes_norm) in enumerate(batch_meta):
            all_cls_tokens.append(cls_tokens[i])
            all_patch_tokens.append(patch_tokens[i])
            all_counts.append(count)
            all_boxes.append(boxes_norm)

    for image_id, meta in tqdm(image_index.items(), desc=split_name, ncols=100):
        image_path = IMAGES_PATH / split_name / meta["file_name"]
        if not image_path.exists():
            continue

        img   = io.read_image(str(image_path))
        img_w = meta["width"]
        img_h = meta["height"]

        raw_boxes = meta["boxes"][:MAX_DET]
        raw_boxes.sort(key=lambda b: (b[1], b[0]))  # sort top-left → bottom-right

        boxes_norm = [_normalize_box(b, img_w, img_h) for b in raw_boxes]
        count      = len(boxes_norm)
        boxes_norm += [[0.0, 0.0, 0.0, 0.0]] * (MAX_DET - count)  # pad unused slots

        batch_imgs.append(_preprocess(img))
        batch_meta.append((count, boxes_norm))

        if len(batch_imgs) == config.batch_size:
            _flush()
            batch_imgs.clear()
            batch_meta.clear()

    if batch_imgs:
        _flush()

    if not all_cls_tokens:
        print(f"[WARNING] No features for {split_name}")
        return

    torch.save(
        {
            "cls_tokens":   torch.stack(all_cls_tokens),                    # (N, D)
            "patch_tokens": torch.stack(all_patch_tokens),                  # (N, N_patches, D)
            "counts":       torch.tensor(all_counts, dtype=torch.long),     # (N,)
            "boxes":        torch.tensor(all_boxes,  dtype=torch.float32),  # (N, MAX_DET, 4)
        },
        save_path,
    )
    print(f"[OK] {split_name}: {len(all_cls_tokens)} images")


def build_detector_features():
    """Precompute DINOv2 CLS tokens + sorted GT boxes for the count+box detector."""
    print("[INFO] Clearing old detector features...")
    if DETECTOR_FEAT_PATH.exists():
        shutil.rmtree(DETECTOR_FEAT_PATH)
    DETECTOR_FEAT_PATH.mkdir(parents=True, exist_ok=True)

    encoder = DINOv2Encoder(model_name=config.dinov2_model, freeze=True).to(config.device)
    encoder.eval()

    _process_detector_split("train", TRAIN_ANNOTATIONS, encoder, DETECTOR_TRAIN_FEAT)
    _process_detector_split("val",   VAL_ANNOTATIONS,   encoder, DETECTOR_VAL_FEAT)
    _process_detector_split("test",  TEST_ANNOTATIONS,  encoder, DETECTOR_TEST_FEAT)

    print("\n[DONE]")
