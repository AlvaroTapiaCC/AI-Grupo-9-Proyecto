import shutil

from torch.utils.data import DataLoader

from ..paths import LAST_METRICS_PATH, BEST_METRICS_PATH, LABEL_ENCODER_PATH

from ..data.label_encoder import LabelEncoder
from ..results.plots import plot_and_save_confusion_matrix
from ..utils.io import save_metrics
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

    plot_and_save_confusion_matrix(
        y_true,
        y_pred,
        save_path=LAST_METRICS_PATH / "confusion_matrix.png",
        class_names=list(label_encoder.id2idx.keys())
    )
    if is_better:
        shutil.copy(LAST_METRICS_PATH / "confusion_matrix.png", BEST_METRICS_PATH / "confusion_matrix.png")

    return metrics