import random
import shutil
import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    DETECTOR_VAL_FEAT,
    VAL_ANNOTATIONS, VAL_IMAGES,
    DET_LAST_RESULTS, DET_BEST_RESULTS,
    DET_LAST_LOGS, DET_BEST_LOGS,
)
from ...data.data_utils import build_image_mapping
from ...data.datasets.detection_dataset import DetectionFeatureDataset
from ...data.datasets.detection_image_dataset import DetectionImageDataset
from ...training.metrics import compute_detector_metrics
from ...utils.box_ops import cxcywh_to_xyxy
from ...utils.io import save_json, load_json
from ...visualization.plots import (
    draw_detector_comparison,
    plot_detector_history,
    plot_count_error_distribution,
    plot_iou_distribution,
)


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def evaluate_detector(model, is_better, dino_encoder=None):
    if config.finetune_dino and dino_encoder is not None:
        val_dataset = DetectionImageDataset(VAL_ANNOTATIONS, VAL_IMAGES)
        val_loader  = DataLoader(val_dataset, batch_size=config.batch_size,
                                 shuffle=False, num_workers=0)
        all_count_logits, all_box_preds      = [], []
        all_count_targets, all_box_targets   = [], []

        dino_encoder.eval()
        model.eval()
        with torch.no_grad():
            for images, count_targets, box_targets in val_loader:
                images        = images.to(config.device)
                cls_tokens, patch_tokens = dino_encoder.forward_detector(images)
                count_logits, box_preds  = model(cls_tokens, patch_tokens)
                all_count_logits.append(count_logits.cpu())
                all_box_preds.append(box_preds.cpu())
                all_count_targets.append(count_targets)
                all_box_targets.append(box_targets)
    else:
        val_dataset = DetectionFeatureDataset(DETECTOR_VAL_FEAT)
        val_loader  = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        model.eval()
        all_count_logits, all_box_preds      = [], []
        all_count_targets, all_box_targets   = [], []

        with torch.no_grad():
            for cls_tokens, patch_tokens, count_targets, box_targets in val_loader:
                count_logits, box_preds = model(
                    cls_tokens.to(config.device), patch_tokens.to(config.device)
                )
                all_count_logits.append(count_logits.cpu())
                all_box_preds.append(box_preds.cpu())
                all_count_targets.append(count_targets)
                all_box_targets.append(box_targets)

    count_logits_cat = torch.cat(all_count_logits)
    box_preds_cat    = torch.cat(all_box_preds)
    count_targets_cat = torch.cat(all_count_targets)
    box_targets_cat   = torch.cat(all_box_targets)

    metrics = compute_detector_metrics(
        count_logits_cat, box_preds_cat,
        count_targets_cat, box_targets_cat,
    )

    results_dir = DET_LAST_RESULTS if config.train_new else DET_BEST_RESULTS
    logs_dir    = DET_LAST_LOGS    if config.train_new else DET_BEST_LOGS

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    for f in results_dir.glob("*.png"):
        f.unlink()
    save_json(logs_dir / "val_metrics.json", metrics)

    # ── Distribution plots ────────────────────────────────────────────────────
    pred_counts = count_logits_cat.argmax(dim=1).tolist()
    gt_counts   = count_targets_cat.tolist()
    plot_count_error_distribution(pred_counts, gt_counts,
                                  results_dir / "count_error_dist.png")

    # Per-slot IoU for all valid slots (cxcywh → xyxy)
    B, MAX_DET, _ = box_preds_cat.shape
    slot_idx = torch.arange(MAX_DET).unsqueeze(0)
    mask     = slot_idx < count_targets_cat.unsqueeze(1)
    if mask.any():
        p = cxcywh_to_xyxy(box_preds_cat[mask])
        t = cxcywh_to_xyxy(box_targets_cat[mask])
        inter = (torch.min(p[:, 2], t[:, 2]) - torch.max(p[:, 0], t[:, 0])).clamp(0) * \
                (torch.min(p[:, 3], t[:, 3]) - torch.max(p[:, 1], t[:, 1])).clamp(0)
        p_area = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
        t_area = (t[:, 2] - t[:, 0]).clamp(0) * (t[:, 3] - t[:, 1]).clamp(0)
        ious   = (inter / (p_area + t_area - inter + 1e-6)).tolist()
        plot_iou_distribution(ious, results_dir / "iou_dist.png")

    # ── Training history plot ─────────────────────────────────────────────────
    history_path = (DET_LAST_LOGS if config.train_new else DET_BEST_LOGS) / "history.json"
    if history_path.exists():
        plot_detector_history(load_json(history_path), results_dir)

    print("[INFO] Detector val metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    if config.train_new and is_better:
        if DET_BEST_RESULTS.exists():
            shutil.rmtree(DET_BEST_RESULTS)
        shutil.copytree(DET_LAST_RESULTS, DET_BEST_RESULTS)
        print("[INFO] Best detector results updated.")

    return metrics


def visualize_detector_predictions(model, dino_encoder, save_dir, n_images=4):
    """Sample val images, run detector, save GT vs predicted box comparisons."""
    ann_data  = load_json(VAL_ANNOTATIONS)
    image_map = build_image_mapping(ann_data["images"])
    img_meta  = {img["id"]: img for img in ann_data["images"]}

    by_image = {}
    for ann in ann_data["annotations"]:
        iid = ann["image_id"]
        if iid not in by_image:
            meta = img_meta[iid]
            by_image[iid] = {
                "file_name": meta["file_name"],
                "width": meta["width"], "height": meta["height"],
                "boxes": [],
            }
        x, y, w, h = ann["bbox"]
        W, H = by_image[iid]["width"], by_image[iid]["height"]
        by_image[iid]["boxes"].append([
            max(0.0, x / W), max(0.0, y / H),
            min(1.0, (x + w) / W), min(1.0, (y + h) / H),
        ])

    image_ids = random.sample(list(by_image.keys()), min(n_images, len(by_image)))
    dino_encoder.eval()
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, iid in enumerate(image_ids):
        meta = by_image[iid]
        image_path = VAL_IMAGES / meta["file_name"]
        if not image_path.exists():
            continue

        img_tensor = io.read_image(str(image_path))
        img_input  = TF.resize(img_tensor, [224, 224], antialias=True)
        img_input  = (img_input.float() / 255.0 - _MEAN) / _STD

        with torch.no_grad():
            cls_token, patch_tokens = dino_encoder.forward_detector(
                img_input.unsqueeze(0).to(config.device)
            )
            count_logits, box_preds = model(cls_token, patch_tokens)

        pred_count = count_logits.argmax(dim=1).item()
        # convert cxcywh → xyxy for visualization
        pred_boxes_xyxy = cxcywh_to_xyxy(box_preds[0, :pred_count].cpu()).tolist()

        draw_detector_comparison(
            image_path,
            gt_boxes=meta["boxes"], pred_boxes=pred_boxes_xyxy,
            gt_count=len(meta["boxes"]), pred_count=pred_count,
            save_path=save_dir / f"det_comparison_{i:02d}.png",
        )
