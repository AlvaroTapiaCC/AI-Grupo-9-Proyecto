import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from ...utils.io import load_json

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MAX_DET = 10


def _normalize_box(box, img_w, img_h):
    """[x, y, w, h] COCO → [cx, cy, w, h] normalized to [0, 1]."""
    x, y, w, h = box
    return [
        max(0.0, min(1.0, (x + w / 2) / img_w)),
        max(0.0, min(1.0, (y + h / 2) / img_h)),
        max(0.0, min(1.0, w / img_w)),
        max(0.0, min(1.0, h / img_h)),
    ]


class DetectionImageDataset(Dataset):
    """
    Loads raw images from disk and returns DINOv2-normalized tensors.
    Used for fine-tuning: the encoder runs inside the training loop.
    """

    def __init__(self, ann_path, images_path):
        data      = load_json(ann_path)
        image_map = {img["id"]: img for img in data["images"]}

        index = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            if iid not in index:
                img = image_map[iid]
                index[iid] = {
                    "file_name": img["file_name"],
                    "width":     img["width"],
                    "height":    img["height"],
                    "boxes":     [],
                }
            index[iid]["boxes"].append(ann["bbox"])

        self.images_path = images_path
        self.samples = []
        for meta in index.values():
            raw = meta["boxes"][:MAX_DET]
            raw.sort(key=lambda b: (b[1], b[0]))
            boxes_norm = [_normalize_box(b, meta["width"], meta["height"]) for b in raw]
            count      = len(boxes_norm)
            boxes_norm += [[0.0, 0.0, 0.0, 0.0]] * (MAX_DET - count)
            self.samples.append({
                "file_name": meta["file_name"],
                "count":     count,
                "boxes":     boxes_norm,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s          = self.samples[idx]
        img_path   = self.images_path / s["file_name"]
        img        = io.read_image(str(img_path))
        img        = TF.resize(img, [224, 224], antialias=True)
        img        = (img.float() / 255.0 - _MEAN) / _STD
        count      = torch.tensor(s["count"], dtype=torch.long)
        boxes      = torch.tensor(s["boxes"], dtype=torch.float32)
        return img, count, boxes
