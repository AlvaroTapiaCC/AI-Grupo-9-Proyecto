import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ── Classifier metrics ────────────────────────────────────────────────────────

def get_classifier_predictions(model, loader, device):
    """Run model over loader, return (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device).float())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def compute_classifier_metrics(y_true, y_pred, average="macro"):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=0),
    }


# ── Detector metrics ──────────────────────────────────────────────────────────

def compute_detector_metrics(count_logits, box_preds, count_targets, box_targets):
    """
    count_logits:  (B, num_count_classes)
    box_preds:     (B, MAX_DET, 4)
    count_targets: (B,)
    box_targets:   (B, MAX_DET, 4)

    Returns dict with count_mae and mean_iou (over valid slots only).
    """
    pred_counts = count_logits.argmax(dim=1)
    count_mae   = (pred_counts - count_targets).abs().float().mean().item()

    B, MAX_DET, _ = box_preds.shape
    slot_idx = torch.arange(MAX_DET, device=box_preds.device).unsqueeze(0)
    mask     = slot_idx < count_targets.unsqueeze(1)  # (B, MAX_DET)

    if not mask.any():
        return {"count_mae": count_mae, "mean_iou": 0.0}

    p = box_preds[mask]    # (N, 4)
    t = box_targets[mask]  # (N, 4)

    inter = (torch.min(p[:, 2], t[:, 2]) - torch.max(p[:, 0], t[:, 0])).clamp(0) * \
            (torch.min(p[:, 3], t[:, 3]) - torch.max(p[:, 1], t[:, 1])).clamp(0)
    p_area = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    t_area = (t[:, 2] - t[:, 0]).clamp(0) * (t[:, 3] - t[:, 1]).clamp(0)
    union  = p_area + t_area - inter + 1e-6
    mean_iou = (inter / union).mean().item()

    return {"count_mae": count_mae, "mean_iou": mean_iou}
