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
    plt.plot(epochs, history["train_count_loss"], label="Train")
    plt.plot(epochs, history["test_count_loss"],  label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Count Loss"); plt.legend(); plt.grid(True)
    plt.savefig(save_path / "count_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_box_loss"], label="Train")
    plt.plot(epochs, history["test_box_loss"],  label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Box Loss"); plt.legend(); plt.grid(True)
    plt.savefig(save_path / "box_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_count_mae"], label="Train")
    plt.plot(epochs, history["test_count_mae"],  label="Val")
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
    plt.xlabel("IoU (predicción vs GT)")
    plt.ylabel("Cantidad de cajas")
    plt.title("Distribución de IoU por caja predicha")
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

def _box_iou_norm(a, b):
    """IoU between two [x1,y1,x2,y2] normalized boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (a_area + b_area - inter + 1e-6)


def _crop(img, box, H, W, pad=4):
    """Crop PIL/numpy image to normalized [x1,y1,x2,y2] box with optional padding."""
    x1 = max(0, int(box[0] * W) - pad)
    y1 = max(0, int(box[1] * H) - pad)
    x2 = min(W, int(box[2] * W) + pad)
    y2 = min(H, int(box[3] * H) + pad)
    return img[y1:y2, x1:x2]


def draw_pipeline_result(image_path, detections, save_path, gt=None, iou_threshold=0.3):
    """
    Per-product grid (as square as possible): one cell per GT product.
    Each cell shows the full image with GT bbox (blue) and matched pred bbox (red).
    If no prediction matched, only the GT bbox is drawn.
    """
    img = np.array(Image.open(str(image_path)).convert("RGB"))
    H, W = img.shape[:2]

    if gt is None or len(gt["boxes"]) == 0:
        return

    gt_boxes   = gt["boxes"]
    gt_classes = gt["classes"]
    n_gt       = len(gt_boxes)

    # Greedy match: for each GT find best-IoU prediction (no reuse)
    matched_pred = [False] * len(detections)
    matches = []  # list of (pred_idx or None)
    for gt_box in gt_boxes:
        best_iou, best_idx = 0.0, -1
        for j, det in enumerate(detections):
            if matched_pred[j]:
                continue
            iou = _box_iou_norm(gt_box, det["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_pred[best_idx] = True
            matches.append(best_idx)
        else:
            matches.append(None)

    import math
    cols = math.ceil(math.sqrt(n_gt))
    rows = math.ceil(n_gt / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes_flat = np.array(axes).flatten() if n_gt > 1 else [axes]

    for i, (gt_box, gt_cls, pred_idx) in enumerate(zip(gt_boxes, gt_classes, matches)):
        ax = axes_flat[i]
        ax.imshow(img)

        # GT bbox — blue
        x1, y1, x2, y2 = gt_box
        ax.add_patch(patches.Rectangle(
            (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
            linewidth=2, edgecolor="#1E88E5", facecolor="none",
        ))

        if pred_idx is not None:
            det = detections[pred_idx]
            px1, py1, px2, py2 = det["bbox"]
            ax.add_patch(patches.Rectangle(
                (px1 * W, py1 * H), (px2 - px1) * W, (py2 - py1) * H,
                linewidth=2, edgecolor="#E53935", facecolor="none",
            ))
            iou = _box_iou_norm(gt_box, det["bbox"])
            label = f"GT: {gt_cls}\nPred: {det['class_name']} (IoU: {iou:.0%})"
        else:
            label = f"GT: {gt_cls}\nNo detectado"

        ax.set_title(label, fontsize=7, loc="left", pad=3)
        ax.axis("off")

    # hide unused subplots
    for j in range(n_gt, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle(image_path.name, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
