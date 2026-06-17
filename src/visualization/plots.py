import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image


_CLASS_COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
    "#00ACC1", "#FFB300", "#6D4C41", "#546E7A", "#D81B60",
]


# ── Classifier ────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def plot_classifier_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"],  label="Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Classifier Loss"); plt.legend(); plt.grid(True)
    plt.savefig(save_path / "loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"],  label="Test Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy"); plt.legend(); plt.grid(True)
    plt.savefig(save_path / "accuracy.png", dpi=150)
    plt.close()


# ── Detector ──────────────────────────────────────────────────────────────────

def plot_detector_history(history, save_path):
    epochs = range(1, len(history["train_count_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_count_loss"], label="Train Count Loss")
    plt.plot(epochs, history["test_count_loss"],  label="Test Count Loss")
    plt.plot(epochs, history["train_box_loss"],   label="Train Box Loss", linestyle="--")
    plt.plot(epochs, history["test_box_loss"],    label="Test Box Loss",  linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Detector Loss"); plt.legend(); plt.grid(True)
    plt.savefig(save_path / "loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_count_mae"], label="Train Count MAE")
    plt.plot(epochs, history["test_count_mae"],  label="Test Count MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE")
    plt.title("Count MAE"); plt.legend(); plt.grid(True)
    plt.savefig(save_path / "count_mae.png", dpi=150)
    plt.close()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def draw_pipeline_result(image_path, detections, save_path):
    """Image with predicted bboxes color-coded by class and confidence."""
    img = np.array(Image.open(str(image_path)).convert("RGB"))
    H, W = img.shape[:2]

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    class_color_map, ci = {}, 0
    for det in detections:
        name = det["class_name"]
        if name not in class_color_map:
            class_color_map[name] = _CLASS_COLORS[ci % len(_CLASS_COLORS)]
            ci += 1
        color = class_color_map[name]
        x1, y1, x2, y2 = det["bbox"]
        ax.add_patch(patches.Rectangle(
            (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        ax.text(
            x1 * W, max(0, y1 * H - 4),
            f"{name} {det['confidence']:.0%}",
            color="white", fontsize=7,
            bbox=dict(facecolor=color, alpha=0.8, pad=1),
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
