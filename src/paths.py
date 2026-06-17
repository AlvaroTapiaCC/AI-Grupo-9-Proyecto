from pathlib import Path
from .config import level

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH         = PROJECT_ROOT / "dataset"
ANNOTATIONS_PATH     = DATASET_PATH / "annotations" / level
IMAGES_PATH          = DATASET_PATH / "retail_product_checkout" / level
CATEGORIES_PATH      = DATASET_PATH / "annotations" / "categories.json"
SUPERCATEGORIES_PATH = DATASET_PATH / "annotations" / "supercategories.json"

TRAIN_ANNOTATIONS = ANNOTATIONS_PATH / "train_annotations.json"
VAL_ANNOTATIONS   = ANNOTATIONS_PATH / "val_annotations.json"
TEST_ANNOTATIONS  = ANNOTATIONS_PATH / "test_annotations.json"

TRAIN_IMAGES = IMAGES_PATH / "train"
VAL_IMAGES   = IMAGES_PATH / "val"
TEST_IMAGES  = IMAGES_PATH / "test"

# ── Precomputed features ──────────────────────────────────────────────────────
# CLIP crop embeddings (for classifier)
CLIP_EMB_PATH      = PROJECT_ROOT / "embeddings" / level
CLIP_LABEL_ENCODER = CLIP_EMB_PATH / "label_encoder.json"
CLIP_TRAIN_EMB     = CLIP_EMB_PATH / "train.pt"
CLIP_VAL_EMB       = CLIP_EMB_PATH / "val.pt"
CLIP_TEST_EMB      = CLIP_EMB_PATH / "test.pt"

# DINOv2 crop embeddings (for classifier comparison)
DINO_EMB_PATH      = PROJECT_ROOT / "dinov2_embeddings" / level
DINO_LABEL_ENCODER = DINO_EMB_PATH / "label_encoder.json"
DINO_TRAIN_EMB     = DINO_EMB_PATH / "train.pt"
DINO_VAL_EMB       = DINO_EMB_PATH / "val.pt"
DINO_TEST_EMB      = DINO_EMB_PATH / "test.pt"

# DINOv2 CLS token features (for detector)
DETECTOR_FEAT_PATH  = PROJECT_ROOT / "detector_features" / level
DETECTOR_TRAIN_FEAT = DETECTOR_FEAT_PATH / "train.pt"
DETECTOR_VAL_FEAT   = DETECTOR_FEAT_PATH / "val.pt"
DETECTOR_TEST_FEAT  = DETECTOR_FEAT_PATH / "test.pt"

# ── Model outputs ─────────────────────────────────────────────────────────────
def _result_paths(name: str) -> dict:
    base = PROJECT_ROOT / "results" / name / level
    return {
        "last_model":   base / "last" / "model",
        "best_model":   base / "best" / "model",
        "last_metrics": base / "last" / "metrics",
        "best_metrics": base / "best" / "metrics",
    }

_CLS = _result_paths("classifier")
CLS_LAST_MODEL   = _CLS["last_model"]
CLS_BEST_MODEL   = _CLS["best_model"]
CLS_LAST_METRICS = _CLS["last_metrics"]
CLS_BEST_METRICS = _CLS["best_metrics"]

_DET = _result_paths("detector")
DET_LAST_MODEL   = _DET["last_model"]
DET_BEST_MODEL   = _DET["best_model"]
DET_LAST_METRICS = _DET["last_metrics"]
DET_BEST_METRICS = _DET["best_metrics"]

# ── Pipeline results ──────────────────────────────────────────────────────────
PIPELINE_RESULTS = PROJECT_ROOT / "results" / "pipeline" / level
