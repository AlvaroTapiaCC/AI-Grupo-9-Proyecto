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
)
from ...data.data_utils import build_image_mapping
from ...data.datasets.detection_dataset import DetectionFeatureDataset
from ...training.metrics import compute_detector_metrics
from ...utils.io import save_json, load_json
from ...visualization.plots import draw_detector_comparison


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def evaluate_detector(model, is_better, val_feat=None):
    val_feat    = val_feat or DETECTOR_VAL_FEAT
    val_dataset = DetectionFeatureDataset(val_feat)
    val_loader  = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model.eval()
    all_count_logits, all_box_preds     = [], []
    all_count_targets, all_box_targets  = [], []

    with torch.no_grad():
        for cls_tokens, count_targets, box_targets in val_loader:
            count_logits, box_preds = model(cls_tokens.to(config.device))
            all_count_logits.append(count_logits.cpu())
            all_box_preds.append(box_preds.cpu())
            all_count_targets.append(count_targets)
            all_box_targets.append(box_targets)

    metrics = compute_detector_metrics(
        torch.cat(all_count_logits),
        torch.cat(all_box_preds),
        torch.cat(all_count_targets),
        torch.cat(all_box_targets),
    )

    DET_LAST_RESULTS.mkdir(parents=True, exist_ok=True)
    save_json(DET_LAST_RESULTS / "val_metrics.json", metrics)

    print("[INFO] Detector val metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    if is_better:
        DET_BEST_RESULTS.mkdir(parents=True, exist_ok=True)
        shutil.copy(DET_LAST_RESULTS / "val_metrics.json", DET_BEST_RESULTS / "val_metrics.json")

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
            cls_token                = dino_encoder(img_input.unsqueeze(0).to(config.device))
            count_logits, box_preds  = model(cls_token)

        pred_count = count_logits.argmax(dim=1).item()
        pred_boxes = box_preds[0, :pred_count].cpu().tolist()

        draw_detector_comparison(
            image_path,
            gt_boxes=meta["boxes"], pred_boxes=pred_boxes,
            gt_count=len(meta["boxes"]), pred_count=pred_count,
            save_path=save_dir / f"det_comparison_{i:02d}.png",
        )
