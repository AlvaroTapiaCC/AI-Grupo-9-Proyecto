import torch
from torch.utils.data import DataLoader, TensorDataset

from ..paths import EMBEDDINGS_PATH, RESULTS_PATH
from ..data.label_encoder import LabelEncoder
from ..utils.io import save_metrics
from .metrics import (
    get_predictions,
    compute_all_metrics,
    plot_and_save_confusion_matrix,
)


def load_embeddings(path):
    data = torch.load(path)
    return TensorDataset(data["embeddings"], data["labels"])


def evaluate_mlp(model, val_path, device, batch_size=32):

    val_dataset = load_embeddings(val_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    label_encoder = LabelEncoder.load(
        EMBEDDINGS_PATH / "label_encoder.json"
    )

    y_true, y_pred = get_predictions(model, val_loader, device)

    metrics = compute_all_metrics(y_true, y_pred)

    save_metrics(metrics, RESULTS_PATH / "metrics.json")

    plot_and_save_confusion_matrix(
        y_true,
        y_pred,
        save_path=RESULTS_PATH / "confusion_matrix.png",
        class_names=list(label_encoder.id2idx.keys())
    )

    return metrics