import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.box_ops import cxcywh_to_xyxy


def _giou_flat(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """GIoU for (N, 4) paired boxes in xyxy format. Returns (N,) GIoU values."""
    px1, py1, px2, py2 = pred[:, 0],   pred[:, 1],   pred[:, 2],   pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) * \
            (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    p_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    t_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union  = p_area + t_area - inter + 1e-6
    iou    = inter / union

    enc_x1 = torch.min(px1, tx1); enc_y1 = torch.min(py1, ty1)
    enc_x2 = torch.max(px2, tx2); enc_y2 = torch.max(py2, ty2)
    enclosing = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + 1e-6

    return iou - (enclosing - union) / enclosing


def _batch_cost_matrix(box_preds: torch.Tensor, box_targets: torch.Tensor) -> np.ndarray:
    """
    Compute (B, MAX_DET, MAX_DET) cost matrix fully on GPU, return as numpy.
    Boxes in cxcywh format — converted to xyxy internally for GIoU.
    """
    B, M, _ = box_preds.shape

    pred_exp = box_preds.unsqueeze(2).expand(B, M, M, 4)
    gt_exp   = box_targets.unsqueeze(1).expand(B, M, M, 4)

    l1_cost = (pred_exp - gt_exp).abs().sum(-1)  # (B, M, M)

    p_flat    = cxcywh_to_xyxy(pred_exp.reshape(B * M * M, 4))
    g_flat    = cxcywh_to_xyxy(gt_exp.reshape(B * M * M, 4))
    giou_cost = (1.0 - _giou_flat(p_flat, g_flat)).reshape(B, M, M)

    return (l1_cost + giou_cost).detach().cpu().numpy()


def detector_loss(count_logits, box_preds, count_targets, box_targets):
    """
    Boxes in cxcywh format.
    Hungarian matching per image — cost matrix in one GPU pass.
    Returns: (count_loss, box_loss)
    """
    count_loss = F.cross_entropy(count_logits, count_targets)

    cost_np = _batch_cost_matrix(box_preds, box_targets)

    box_losses = []
    for b in range(box_preds.shape[0]):
        n = count_targets[b].item()
        if n == 0:
            continue

        row_idx, col_idx = linear_sum_assignment(cost_np[b, :, :n])

        matched_pred     = box_preds[b][row_idx]
        matched_gt       = box_targets[b][col_idx]
        matched_pred_xy  = cxcywh_to_xyxy(matched_pred)
        matched_gt_xy    = cxcywh_to_xyxy(matched_gt)

        box_losses.append(
            F.smooth_l1_loss(matched_pred, matched_gt) +
            (1.0 - _giou_flat(matched_pred_xy, matched_gt_xy)).mean()
        )

    if not box_losses:
        return count_loss, torch.tensor(0.0, device=box_preds.device, requires_grad=True)

    return count_loss, torch.stack(box_losses).mean()
