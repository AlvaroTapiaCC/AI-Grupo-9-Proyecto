from torch.utils.data import DataLoader

from ..paths import VAL_ANNOTATIONS, EMBEDDINGS_PATH, LAST_METRICS_PATH

from ..data.dataset_loader import RetailDataset
from ..data.label_encoder import LabelEncoder
from ..data.transforms import get_val_transforms
from ..utils.io import save_metrics
from .metrics import (
    get_predictions,
    compute_all_metrics,
    plot_and_save_confusion_matrix,
)


def evaluate_cnn(model, device, batch_size=32):

    val_dataset = RetailDataset(
        VAL_ANNOTATIONS,
        split="val",
        transform=get_val_transforms(),
        label_encoder_path=EMBEDDINGS_PATH / "label_encoder.json"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    label_encoder = LabelEncoder.load(
        EMBEDDINGS_PATH / "label_encoder.json"
    )

    y_true, y_pred = get_predictions(model, val_loader, device)

    metrics = compute_all_metrics(y_true, y_pred)

    save_metrics(metrics, LAST_METRICS_PATH / "metrics.json")

    plot_and_save_confusion_matrix(
        y_true,
        y_pred,
        save_path=LAST_METRICS_PATH / "confusion_matrix.png",
        class_names=list(label_encoder.id2idx.keys())
    )

    return metrics