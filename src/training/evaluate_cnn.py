import shutil

from torch.utils.data import DataLoader

from ..paths import (
    LAST_METRICS_PATH, BEST_METRICS_PATH, 
    LABEL_ENCODER_PATH,
    VAL_ANNOTATIONS, SUPERCATEGORIES_PATH,
    VAL_IMAGES
    )

from ..data.label_encoder import LabelEncoder
from ..data.transforms import get_val_transforms
from ..data.data_utils import build_supercategory_name_mapping

from ..results.plots import plot_and_save_confusion_matrix
from ..results.class_pred import show_predictions_on_image

from ..utils.io import save_metrics, load_json

from .metrics import (
    get_predictions,
    compute_all_metrics,
)
from .training_utils import load_tensors


def evaluate_cnn(model, is_better, val_path, device, batch_size=32):

    print("[INFO] Loading precomputed val tensors...")

    val_dataset = load_tensors(val_path)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    label_encoder = LabelEncoder.load(LABEL_ENCODER_PATH)

    y_true, y_pred = get_predictions(model, val_loader, device)

    metrics = compute_all_metrics(y_true, y_pred)

    save_metrics(metrics, LAST_METRICS_PATH / "metrics.json")
    
    supercats = load_json(SUPERCATEGORIES_PATH)
    supercat_map = build_supercategory_name_mapping(supercats)
    supercat_list = [name for name in supercat_map.values()]

    cm = plot_and_save_confusion_matrix(
        y_true,
        y_pred,
        save_path=LAST_METRICS_PATH / "confusion_matrix.png",
        class_names=supercat_list
    )
    print("[INFO] Drawing predictions on example image...")
    
    clip_model = None
    preprocess = get_val_transforms()
    
    show_predictions_on_image(model, device, cm, VAL_ANNOTATIONS, VAL_IMAGES, LAST_METRICS_PATH, label_encoder, clip_model, preprocess)
    
    if is_better:
        shutil.copy(LAST_METRICS_PATH / "confusion_matrix.png", BEST_METRICS_PATH / "confusion_matrix.png")
        shutil.copy(LAST_METRICS_PATH / "best_pred_img.png", BEST_METRICS_PATH / "best_pred_img.png")
        shutil.copy(LAST_METRICS_PATH / "worst_pred_img.png", BEST_METRICS_PATH / "worst_pred_img.png")

    return metrics