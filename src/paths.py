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
PRECOMPUTED_PATH = PROJECT_ROOT / "precomputed"

CLIP_EMB_PATH      = PRECOMPUTED_PATH / "clip" / level
CLIP_LABEL_ENCODER = CLIP_EMB_PATH / "label_encoder.json"
CLIP_TRAIN_EMB     = CLIP_EMB_PATH / "train.pt"
CLIP_VAL_EMB       = CLIP_EMB_PATH / "val.pt"
CLIP_TEST_EMB      = CLIP_EMB_PATH / "test.pt"

DINO_EMB_PATH      = PRECOMPUTED_PATH / "dinov2" / level
DINO_LABEL_ENCODER = DINO_EMB_PATH / "label_encoder.json"
DINO_TRAIN_EMB     = DINO_EMB_PATH / "train.pt"
DINO_VAL_EMB       = DINO_EMB_PATH / "val.pt"
DINO_TEST_EMB      = DINO_EMB_PATH / "test.pt"

DETECTOR_FEAT_PATH  = PRECOMPUTED_PATH / "detector" / level
DETECTOR_TRAIN_FEAT = DETECTOR_FEAT_PATH / "train.pt"
DETECTOR_VAL_FEAT   = DETECTOR_FEAT_PATH / "val.pt"
DETECTOR_TEST_FEAT  = DETECTOR_FEAT_PATH / "test.pt"

# ── Checkpoints (model weights) ───────────────────────────────────────────────
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"

CLS_LAST_CHECKPOINT = CHECKPOINTS_PATH / "classifier" / level / "last.pt"
CLS_BEST_CHECKPOINT = CHECKPOINTS_PATH / "classifier" / level / "best.pt"

DET_LAST_CHECKPOINT      = CHECKPOINTS_PATH / "detector" / level / "last.pt"
DET_BEST_CHECKPOINT      = CHECKPOINTS_PATH / "detector" / level / "best.pt"
DINO_DET_LAST_CHECKPOINT = CHECKPOINTS_PATH / "detector" / level / "dino_last.pt"
DINO_DET_BEST_CHECKPOINT = CHECKPOINTS_PATH / "detector" / level / "dino_best.pt"

# ── Results (metrics and plots) ───────────────────────────────────────────────
RESULTS_PATH = PROJECT_ROOT / "results"

CLS_LAST_RESULTS = RESULTS_PATH / "classifier" / level / "last"
CLS_BEST_RESULTS = RESULTS_PATH / "classifier" / level / "best"
CLS_LAST_LOGS    = CLS_LAST_RESULTS / "logs"
CLS_BEST_LOGS    = CLS_BEST_RESULTS / "logs"

DET_LAST_RESULTS = RESULTS_PATH / "detector" / level / "last"
DET_BEST_RESULTS = RESULTS_PATH / "detector" / level / "best"
DET_LAST_LOGS    = DET_LAST_RESULTS / "logs"
DET_BEST_LOGS    = DET_BEST_RESULTS / "logs"

PIPELINE_RESULTS = RESULTS_PATH / "pipeline" / level
