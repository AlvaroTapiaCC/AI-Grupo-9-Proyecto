import random
import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import clip

from ..paths import (
    VAL_ANNOTATIONS, VAL_IMAGES,
    CATEGORIES_PATH, SUPERCATEGORIES_PATH,
)
from ..data.data_utils import (
    build_image_mapping, build_category_mapping, build_supercategory_name_mapping,
)
from ..utils.io import load_json


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _encode_crop(img_tensor, bbox, clip_model, device):
    x, y, w, h = bbox
    _, H, W = img_tensor.shape
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(W, int(x + w)), min(H, int(y + h))
    crop = TF.resize(img_tensor[:, y1:y2, x1:x2], [224, 224], antialias=True)
    crop = (crop.float() / 255.0 - _CLIP_MEAN) / _CLIP_STD
    with torch.no_grad():
        return clip_model.encode_image(crop.unsqueeze(0).to(device))


def show_classifier_predictions(model, label_encoder, device, save_dir, n_images=4):
    """
    Samples n_images from the val set, runs CLIP + MLP on each GT crop,
    and saves an annotated image with green (correct) / red (incorrect) boxes.
    """
    ann_data    = load_json(VAL_ANNOTATIONS)
    cat_map     = build_category_mapping(load_json(CATEGORIES_PATH))
    supercat_map = build_supercategory_name_mapping(load_json(SUPERCATEGORIES_PATH))
    image_map   = build_image_mapping(ann_data["images"])

    # group annotations by image
    by_image = {}
    for ann in ann_data["annotations"]:
        iid = ann["image_id"]
        by_image.setdefault(iid, []).append(ann)

    image_ids = random.sample(list(by_image.keys()), min(n_images, len(by_image)))

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    model.eval()

    for iid in image_ids:
        file_name  = image_map.get(iid)
        image_path = VAL_IMAGES / file_name
        if not image_path.exists():
            continue

        img_tensor = io.read_image(str(image_path))
        _, H, W    = img_tensor.shape
        img_np     = img_tensor.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img_np)

        for ann in by_image[iid]:
            supercat_id = cat_map.get(ann["category_id"])
            if supercat_id is None:
                continue
            true_idx = label_encoder.id2idx.get(supercat_id)
            if true_idx is None:
                continue

            emb      = _encode_crop(img_tensor, ann["bbox"], clip_model, device)
            pred_idx = model(emb).argmax(dim=1).item()
            correct  = pred_idx == true_idx

            x, y, w, h = ann["bbox"]
            color = "#43A047" if correct else "#E53935"
            ax.add_patch(patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor=color, facecolor="none",
            ))

            true_name = supercat_map.get(supercat_id, str(supercat_id))
            pred_name = supercat_map.get(label_encoder.idx2id.get(pred_idx), str(pred_idx))
            label = true_name if correct else f"P:{pred_name} T:{true_name}"
            ax.text(
                x, max(0, y - 4), label,
                color="white", fontsize=6,
                bbox=dict(facecolor=color, alpha=0.8, pad=1),
            )

        ax.set_title(f"val/{file_name}", fontsize=8)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_dir / f"pred_{iid}.png", bbox_inches="tight", dpi=150)
        plt.close()
