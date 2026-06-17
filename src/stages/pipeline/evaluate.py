import random
from ... import config
from ...paths import (
    VAL_ANNOTATIONS, VAL_IMAGES,
    CATEGORIES_PATH, SUPERCATEGORIES_PATH,
    PIPELINE_RESULTS,
)
from ...data.data_utils import build_image_mapping, build_category_mapping, build_supercategory_name_mapping
from ...utils.io import load_json, save_json
from ...inference.pipeline import run_pipeline
from ...training.metrics import compute_pipeline_metrics
from ...visualization.plots import draw_pipeline_result


def _build_gt(ann_data, cat_map, supercat_map):
    """Returns {image_id: {file_name, boxes:[[x1,y1,x2,y2] norm], classes:[str]}}."""
    img_meta = {img["id"]: img for img in ann_data["images"]}
    by_image = {}
    for ann in ann_data["annotations"]:
        iid = ann["image_id"]
        supercat_id = cat_map.get(ann["category_id"])
        if supercat_id is None:
            continue
        if iid not in by_image:
            meta = img_meta[iid]
            by_image[iid] = {
                "file_name": meta["file_name"],
                "width":     meta["width"],
                "height":    meta["height"],
                "boxes":     [],
                "classes":   [],
            }
        x, y, w, h = ann["bbox"]
        W, H = by_image[iid]["width"], by_image[iid]["height"]
        by_image[iid]["boxes"].append([
            max(0.0, x / W), max(0.0, y / H),
            min(1.0, (x + w) / W), min(1.0, (y + h) / H),
        ])
        by_image[iid]["classes"].append(supercat_map.get(supercat_id, str(supercat_id)))
    return by_image


def evaluate_pipeline(
    dino_encoder, detector, clip_model,
    mlp_classifier, label_encoder, supercat_map,
    n_images=None,
):
    """
    Run full pipeline on val set, compute combined metrics, save sample visualizations.

    Returns dict: loc_recall, clf_accuracy, end_to_end, combined_score.
    """
    ann_data = load_json(VAL_ANNOTATIONS)
    cat_map  = build_category_mapping(load_json(CATEGORIES_PATH))
    supercat_map_loaded = build_supercategory_name_mapping(load_json(SUPERCATEGORIES_PATH))

    by_image  = _build_gt(ann_data, cat_map, supercat_map_loaded)
    image_ids = list(by_image.keys())
    if n_images is not None:
        image_ids = random.sample(image_ids, min(n_images, len(image_ids)))

    PIPELINE_RESULTS.mkdir(parents=True, exist_ok=True)

    all_preds, all_gt = [], []

    print(f"[INFO] Evaluating pipeline on {len(image_ids)} val images...")
    for i, iid in enumerate(image_ids):
        meta       = by_image[iid]
        image_path = VAL_IMAGES / meta["file_name"]
        if not image_path.exists():
            continue

        detections = run_pipeline(
            image_path, dino_encoder, detector,
            clip_model, mlp_classifier, label_encoder,
            supercat_map_loaded, config.device,
        )

        all_preds.append(detections)
        all_gt.append({"boxes": meta["boxes"], "classes": meta["classes"]})

        save_path = PIPELINE_RESULTS / f"result_{i:02d}_{image_path.stem}.png"
        draw_pipeline_result(image_path, detections, save_path)
        print(f"  [{i+1}/{len(image_ids)}] {meta['file_name']} → {len(detections)} detections")

    metrics = compute_pipeline_metrics(all_preds, all_gt)
    save_json(PIPELINE_RESULTS / "metrics.json", metrics)

    print("\n[INFO] Pipeline metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    return metrics
