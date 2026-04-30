import shutil
import clip

from torch.utils.data import DataLoader

from ..paths import (
    EMBEDDINGS_PATH, 
    BEST_METRICS_PATH, LAST_METRICS_PATH,
    VAL_ANNOTATIONS, 
    VAL_IMAGES
    )

from ..data.label_encoder import LabelEncoder
from ..data.dataset_loader import load_embeddings
from ..results.plots import plot_and_save_confusion_matrix
from ..results.class_pred import show_predictions_on_image
from ..utils.io import save_metrics
from .metrics import (
    get_predictions,
    compute_all_metrics,
)


def evaluate_mlp(model, is_better, val_path, device, batch_size):

    val_dataset = load_embeddings(val_path)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    label_encoder = LabelEncoder.load(
        EMBEDDINGS_PATH / "label_encoder.json"
    )

    y_true, y_pred = get_predictions(model, val_loader, device)

    metrics = compute_all_metrics(y_true, y_pred)

    save_metrics(metrics, LAST_METRICS_PATH / "metrics.json")

    cm = plot_and_save_confusion_matrix(
        y_true,
        y_pred,
        save_path=LAST_METRICS_PATH / "confusion_matrix.png",
        class_names=list(label_encoder.id2idx.keys())
    )
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    show_predictions_on_image(model, device, cm, VAL_ANNOTATIONS, VAL_IMAGES, LAST_METRICS_PATH, label_encoder, clip_model, preprocess)
    
    if is_better:
        shutil.copy(LAST_METRICS_PATH / "confusion_matrix.png", BEST_METRICS_PATH / "confusion_matrix.png")
        shutil.copy(LAST_METRICS_PATH / "best_pred_img.png", BEST_METRICS_PATH / "best_pred_img.png")
        shutil.copy(LAST_METRICS_PATH / "worst_pred_img.png", BEST_METRICS_PATH / "worst_pred_img.png")     

    return metrics