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


# ── Pipeline metrics ──────────────────────────────────────────────────────────

def _box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (a_area + b_area - inter + 1e-6)


def compute_pipeline_metrics(pred_list, gt_list, iou_threshold=0.5):
    """
    pred_list: list over images of [{bbox:[x1,y1,x2,y2], class_name, confidence}]
    gt_list:   list over images of {boxes:[[x1,y1,x2,y2]], classes:[str]}

    Greedy matching: each GT box matched to its best-IoU prediction (no reuse).

    Returns loc_recall, clf_accuracy, end_to_end, combined_score.
    """
    total_gt = total_detected = total_correct = 0

    for preds, gt in zip(pred_list, gt_list):
        gt_boxes, gt_classes = gt["boxes"], gt["classes"]
        total_gt += len(gt_boxes)
        matched = [False] * len(preds)

        for gt_box, gt_cls in zip(gt_boxes, gt_classes):
            best_iou, best_idx = 0.0, -1
            for j, det in enumerate(preds):
                if matched[j]:
                    continue
                iou = _box_iou(gt_box, det["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, j

            if best_iou >= iou_threshold and best_idx >= 0:
                total_detected += 1
                matched[best_idx] = True
                if preds[best_idx]["class_name"] == gt_cls:
                    total_correct += 1

    loc_recall   = total_detected / max(total_gt, 1)
    clf_accuracy = total_correct  / max(total_detected, 1)
    end_to_end   = total_correct  / max(total_gt, 1)

    return {
        "loc_recall":     round(loc_recall,   4),
        "clf_accuracy":   round(clf_accuracy,  4),
        "end_to_end":     round(end_to_end,    4),
        "combined_score": round((loc_recall + clf_accuracy + end_to_end) / 3, 4),
        "total_gt":       total_gt,
        "total_detected": total_detected,
        "total_correct":  total_correct,
    }
