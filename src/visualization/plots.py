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


# ── Detector ─────────────────────────────────────────────────────────────────

def draw_detector_comparison(image_path, gt_boxes, pred_boxes, gt_count, pred_count, save_path):
    """
    2-panel figure: GT boxes (green) | Predicted boxes (red).
    All boxes in normalized [x1, y1, x2, y2].
    """
    img = np.array(Image.open(str(image_path)).convert("RGB"))
    H, W = img.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, boxes, color, title in [
        (axes[0], gt_boxes,   "#43A047", f"GT ({gt_count} objects)"),
        (axes[1], pred_boxes, "#E53935", f"Predicted ({pred_count} objects)"),
    ]:
        ax.imshow(img)
        for box in boxes:
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                linewidth=2, edgecolor=color, facecolor="none",
            ))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def draw_pipeline_result(image_path, detections, save_path, gt=None):
    """
    Two-panel figure: GT (left) | Predicted (right).
    Both panels color-code boxes by class name.
    gt: {boxes: [[x1,y1,x2,y2] norm], classes: [str]} or None (single panel).
    """
    img = np.array(Image.open(str(image_path)).convert("RGB"))
    H, W = img.shape[:2]

    # Build a shared class → color map across GT and predictions
    all_classes = []
    if gt:
        all_classes += gt["classes"]
    all_classes += [d["class_name"] for d in detections]
    class_color_map, ci = {}, 0
    for name in all_classes:
        if name not in class_color_map:
            class_color_map[name] = _CLASS_COLORS[ci % len(_CLASS_COLORS)]
            ci += 1

    n_panels = 2 if gt else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 8))
    if n_panels == 1:
        axes = [axes]

    def _draw_boxes(ax, boxes, classes, title, show_conf=False, confs=None):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        for j, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box
            color = class_color_map.get(cls, "#FFFFFF")
            ax.add_patch(patches.Rectangle(
                (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                linewidth=2, edgecolor=color, facecolor="none",
            ))
            label = f"{cls} {confs[j]:.0%}" if (show_conf and confs) else cls
            ax.text(
                x1 * W, max(0, y1 * H - 4), label,
                color="white", fontsize=7,
                bbox=dict(facecolor=color, alpha=0.8, pad=1),
            )

    if gt:
        _draw_boxes(axes[0], gt["boxes"], gt["classes"],
                    f"GT ({len(gt['boxes'])} objects)")

    pred_boxes   = [d["bbox"]       for d in detections]
    pred_classes = [d["class_name"] for d in detections]
    pred_confs   = [d["confidence"] for d in detections]
    _draw_boxes(axes[-1], pred_boxes, pred_classes,
                f"Predicted ({len(detections)} objects)",
                show_conf=True, confs=pred_confs)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
