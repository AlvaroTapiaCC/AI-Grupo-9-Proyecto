import torch
import torch.nn.functional as F


def _giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """GIoU loss for (N, 4) boxes in [x1, y1, x2, y2] normalized format."""
    px1, py1, px2, py2 = pred[:, 0],   pred[:, 1],   pred[:, 2],   pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) * \
            (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    p_area    = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    t_area    = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union     = p_area + t_area - inter + 1e-6
    iou       = inter / union

    enc_x1, enc_y1 = torch.min(px1, tx1), torch.min(py1, ty1)
    enc_x2, enc_y2 = torch.max(px2, tx2), torch.max(py2, ty2)
    enclosing = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + 1e-6

    giou = iou - (enclosing - union) / enclosing
    return (1.0 - giou).mean()


def detector_loss(count_logits, box_preds, count_targets, box_targets):
    """
    count_logits:  (B, num_count_classes)
    box_preds:     (B, MAX_DET, 4)  sigmoid output in [0, 1]
    count_targets: (B,)             int64, number of objects per image
    box_targets:   (B, MAX_DET, 4)  ground truth, padded with zeros

    Returns: (count_loss, box_loss)
    """
    count_loss = F.cross_entropy(count_logits, count_targets)

    B, MAX_DET, _ = box_preds.shape
    slot_idx = torch.arange(MAX_DET, device=box_preds.device).unsqueeze(0)  # (1, MAX_DET)
    mask     = slot_idx < count_targets.unsqueeze(1)                         # (B, MAX_DET)

    if not mask.any():
        return count_loss, torch.tensor(0.0, device=box_preds.device, requires_grad=True)

    pred_valid = box_preds[mask]    # (N_valid, 4)
    targ_valid = box_targets[mask]  # (N_valid, 4)

    box_loss = F.smooth_l1_loss(pred_valid, targ_valid) + _giou_loss(pred_valid, targ_valid)
    return count_loss, box_loss
