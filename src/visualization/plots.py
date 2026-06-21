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


# ── Latent space (PCA / UMAP) ────────────────────────────────────────────────

def plot_latent_space(embeddings: np.ndarray, labels: np.ndarray,
                      class_names: list, save_dir):
    """PCA and UMAP 2-D scatter of embeddings, colored by class. Saves two PNGs."""
    from sklearn.decomposition import PCA
    import umap as umap_lib

    palette = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00ACC1", "#FFB300", "#6D4C41", "#546E7A", "#D81B60",
        "#00897B", "#F4511E",
    ]

    def _scatter(ax, proj, title):
        for idx, name in enumerate(class_names):
            mask = labels == idx
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=palette[idx % len(palette)], label=name,
                       alpha=0.6, s=12, linewidths=0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=1.5,
                  bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    # PCA
    pca_proj = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    fig, ax  = plt.subplots(figsize=(8, 6))
    _scatter(ax, pca_proj, "Espacio Latente CLIP — PCA 2D")
    plt.tight_layout()
    plt.savefig(save_dir / "latent_pca.png", dpi=150, bbox_inches="tight")
    plt.close()

    # UMAP
    umap_proj = umap_lib.UMAP(n_components=2, n_neighbors=15,
                               min_dist=0.1, random_state=42).fit_transform(embeddings)
    fig, ax   = plt.subplots(figsize=(8, 6))
    _scatter(ax, umap_proj, "Espacio Latente CLIP — UMAP 2D")
    plt.tight_layout()
    plt.savefig(save_dir / "latent_umap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Detector distributions ────────────────────────────────────────────────────

def plot_count_error_distribution(pred_counts, gt_counts, save_path):
    errors = [int(p) - int(g) for p, g in zip(pred_counts, gt_counts)]
    lo, hi = min(errors), max(errors)
    bins   = list(range(lo, hi + 2))
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=bins, align="left", edgecolor="black", color="steelblue")
    plt.xlabel("Count Error (pred − gt)")
    plt.ylabel("Frequency")
    plt.title("Count Error Distribution")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_iou_distribution(ious, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(ious, bins=20, range=(0, 1), edgecolor="black", color="teal")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.title("IoU Distribution")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Pipeline metrics ──────────────────────────────────────────────────────────

def plot_pipeline_metrics(metrics, save_path):
    keys   = ["loc_recall", "clf_accuracy", "end_to_end", "combined_score"]
    labels = ["Loc Recall", "Clf Accuracy", "End-to-End", "Combined"]
    values = [metrics.get(k, 0.0) for k in keys]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=_CLASS_COLORS[:4], edgecolor="black")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.15)
    plt.ylabel("Score")
    plt.title("Pipeline Metrics")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pipeline_class_accuracy(class_results, save_path):
    """class_results: {class_name: {"total": int, "correct": int}}"""
    names  = list(class_results.keys())
    accs   = [class_results[n]["correct"] / max(class_results[n]["total"], 1) for n in names]
    colors = [_CLASS_COLORS[i % len(_CLASS_COLORS)] for i in range(len(names))]
    plt.figure(figsize=(max(8, len(names) * 0.9), 5))
    bars = plt.bar(names, accs, color=colors, edgecolor="black")
    for bar, val in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", fontsize=8)
    plt.ylim(0, 1.15)
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
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
