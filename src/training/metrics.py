import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def get_predictions(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred, average="macro"):
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_true, y_pred, average="macro"):
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def f1(y_true, y_pred, average="macro"):
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    save_path,
    class_names=None
):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else False,
        yticklabels=class_names if class_names is not None else False
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    return cm


def compute_all_metrics(y_true, y_pred, average="macro"):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, average),
        "recall": recall(y_true, y_pred, average),
        "f1": f1(y_true, y_pred, average),
    }