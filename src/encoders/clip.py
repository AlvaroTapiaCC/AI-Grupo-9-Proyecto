import shutil
import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from tqdm import tqdm
import clip

from .. import config
from ..paths import (
    IMAGES_PATH,
    TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, TEST_ANNOTATIONS,
    CATEGORIES_PATH,
    CLIP_EMB_PATH, CLIP_LABEL_ENCODER,
    CLIP_TRAIN_EMB, CLIP_VAL_EMB, CLIP_TEST_EMB,
)
from ..data.label_encoder import LabelEncoder
from ..utils.io import load_json
from ..data.data_utils import build_category_mapping, build_image_mapping


_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

_SPLIT_PATHS = {
    "train": CLIP_TRAIN_EMB,
    "val":   CLIP_VAL_EMB,
    "test":  CLIP_TEST_EMB,
}


class CLIPEncoder:
    def __init__(self, device=None):
        self.device = device or config.device
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def preprocess(self, img_tensor: torch.Tensor, bbox) -> torch.Tensor:
        """Crop (C,H,W) uint8 → (C,224,224) float32 CLIP-normalized."""
        x, y, w, h = bbox
        _, H, W = img_tensor.shape
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        crop = TF.resize(img_tensor[:, y1:y2, x1:x2], [224, 224], antialias=True)
        return (crop.float() / 255.0 - _MEAN) / _STD

    @torch.no_grad()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch.to(self.device)).cpu()


def _process_split(split_name, ann_path, cat_map, label_encoder, encoder):
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

        batch_crops.append(encoder.preprocess(current_image, ann["bbox"]))
        batch_labels.append(supercat_id)

        if len(batch_crops) == config.batch_size:
            all_embs.append(encoder.encode(torch.stack(batch_crops)))
            raw_labels.extend(batch_labels)
            batch_crops, batch_labels = [], []

    if batch_crops:
        all_embs.append(encoder.encode(torch.stack(batch_crops)))
        raw_labels.extend(batch_labels)

    if not all_embs:
        print(f"[WARNING] No embeddings for {split_name}")
        return

    embeddings = torch.cat(all_embs, dim=0)
    labels     = torch.tensor(label_encoder.transform(raw_labels), dtype=torch.long)
    torch.save({"embeddings": embeddings, "labels": labels}, _SPLIT_PATHS[split_name])
    print(f"[OK] {split_name}: {embeddings.shape}")


def build_embeddings():
    print("[INFO] Clearing old CLIP embeddings...")
    if CLIP_EMB_PATH.exists():
        shutil.rmtree(CLIP_EMB_PATH)
    CLIP_EMB_PATH.mkdir(parents=True, exist_ok=True)

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
    label_encoder.save(CLIP_LABEL_ENCODER)

    encoder = CLIPEncoder()
    _process_split("train", TRAIN_ANNOTATIONS, cat_map, label_encoder, encoder)
    _process_split("val",   VAL_ANNOTATIONS,   cat_map, label_encoder, encoder)
    _process_split("test",  TEST_ANNOTATIONS,  cat_map, label_encoder, encoder)

    print("\n[DONE]")
