from pathlib import Path
from .config import level, model

# ALL PATHS GO HERE FOR EASY ACCESS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results" / model / level

DATASET_PATH = PROJECT_ROOT / "dataset" 
ANNOTATIONS_PATH = DATASET_PATH / "annotations" / level
IMAGES_PATH = DATASET_PATH / "retail_product_checkout" / level

TRAIN_ANNOTATIONS = ANNOTATIONS_PATH / "train_annotations.json"
TEST_ANNOTATIONS = ANNOTATIONS_PATH / "test_annotations.json"
VAL_ANNOTATIONS = ANNOTATIONS_PATH / "val_annotations.json"

TRAIN_IMAGES = IMAGES_PATH / "train"
TEST_IMAGES = IMAGES_PATH / "test"
VAL_IMAGES = IMAGES_PATH / "val"

CATEGORIES_PATH = DATASET_PATH / "annotations" / "categories.json"
SUPERCATEGORIES_PATH = DATASET_PATH / "annotations" / "supercategories.json"

EMBEDDINGS_PATH = PROJECT_ROOT/ "embeddings" / level
TRAIN_EMB = EMBEDDINGS_PATH / "train.pt"
VAL_EMB = EMBEDDINGS_PATH / "val.pt"
TEST_EMB = EMBEDDINGS_PATH / "test.pt"

LAST_MODEL_PATH = RESULTS_PATH / "last" / "model"
BEST_MODEL_PATH = RESULTS_PATH / "best" / "model"
LAST_METRICS_PATH = RESULTS_PATH / "last" / "metrics"
BEST_METRICS_PATH = RESULTS_PATH / "best" / "metrics"

LABEL_ENCODER_PATH = EMBEDDINGS_PATH / "label_encoder.json"

TENSORS_PATH = PROJECT_ROOT / "tensors" / level
TRAIN_TENS = TENSORS_PATH / "train.pt"
TEST_TENS = TENSORS_PATH / "test.pt"
VAL_TENS = TENSORS_PATH / "val.pt"

BEST_MLP_HISTORY = PROJECT_ROOT / "results" / "mlp" / level / "best" / "model" / "history.json"
BEST_CNN_HISTORY = PROJECT_ROOT / "results" / "cnn" / level / "best" / "model" / "history.json" 
COMPARISON_PATH = PROJECT_ROOT / "results" / "comparison" / level