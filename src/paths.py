from pathlib import Path
from .config import level, model

# ALL PATHS GO HERE FOR EASY ACCESS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results" / model


DATASET_PATH = PROJECT_ROOT / "dataset" 
ANNOTATIONS_PATH = DATASET_PATH / "annotations"
IMAGES_PATH = DATASET_PATH / "retail_product_checkout" / level

TRAIN_ANNOTATIONS = ANNOTATIONS_PATH / level / "train_annotations.json"
TEST_ANNOTATIONS = ANNOTATIONS_PATH / level / "test_annotations.json"
VAL_ANNOTATIONS = ANNOTATIONS_PATH / level / "val_annotations.json"

CATEGORIES_PATH = ANNOTATIONS_PATH / "categories.json"

EMBEDDINGS_PATH = PROJECT_ROOT/ "embeddings" / level
TRAIN_EMB = EMBEDDINGS_PATH / "train.pt"
VAL_EMB = EMBEDDINGS_PATH / "val.pt"
TEST_EMB = EMBEDDINGS_PATH / "test.pt"

