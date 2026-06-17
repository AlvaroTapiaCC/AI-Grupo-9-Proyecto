import shutil
import torch
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    DETECTOR_VAL_FEAT,
    DET_LAST_METRICS, DET_BEST_METRICS,
)
from ...data.datasets.detection_dataset import DetectionFeatureDataset
from ...training.metrics import compute_detector_metrics
from ...utils.io import save_json


def evaluate_detector(model, is_better, val_feat=None):
    val_feat    = val_feat or DETECTOR_VAL_FEAT
    val_dataset = DetectionFeatureDataset(val_feat)
    val_loader  = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model.eval()
    all_count_logits, all_box_preds = [], []
    all_count_targets, all_box_targets = [], []

    with torch.no_grad():
        for cls_tokens, count_targets, box_targets in val_loader:
            cls_tokens    = cls_tokens.to(config.device)
            count_logits, box_preds = model(cls_tokens)
            all_count_logits.append(count_logits.cpu())
            all_box_preds.append(box_preds.cpu())
            all_count_targets.append(count_targets)
            all_box_targets.append(box_targets)

    count_logits  = torch.cat(all_count_logits)
    box_preds     = torch.cat(all_box_preds)
    count_targets = torch.cat(all_count_targets)
    box_targets   = torch.cat(all_box_targets)

    metrics = compute_detector_metrics(count_logits, box_preds, count_targets, box_targets)

    DET_LAST_METRICS.mkdir(parents=True, exist_ok=True)
    save_json(DET_LAST_METRICS / "val_metrics.json", metrics)

    print("[INFO] Detector val metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    if is_better:
        DET_BEST_METRICS.mkdir(parents=True, exist_ok=True)
        shutil.copy(DET_LAST_METRICS / "val_metrics.json", DET_BEST_METRICS / "val_metrics.json")

    return metrics
