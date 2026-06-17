import torch
import torchvision.io as io
import torchvision.transforms.functional as TF
from PIL import Image


_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _dino_preprocess(image_tensor: torch.Tensor) -> torch.Tensor:
    """Full image (C,H,W) uint8 → (1,C,224,224) float32 DINOv2-normalized."""
    img = TF.resize(image_tensor, [224, 224], antialias=True)
    img = (img.float() / 255.0 - _DINO_MEAN) / _DINO_STD
    return img.unsqueeze(0)


def _clip_crop(pil_image: Image.Image, box_norm, W, H) -> torch.Tensor:
    """Normalized [x1,y1,x2,y2] → CLIP-preprocessed crop tensor (1,C,224,224)."""
    x1, y1, x2, y2 = box_norm
    px1 = max(0, int(x1 * W)); py1 = max(0, int(y1 * H))
    px2 = min(W, int(x2 * W)); py2 = min(H, int(y2 * H))
    crop = pil_image.crop((px1, py1, px2, py2))
    t    = TF.to_tensor(crop)                                    # [0,1]
    t    = TF.resize(t, [224, 224], antialias=True)
    t    = (t - _CLIP_MEAN) / _CLIP_STD
    return t.unsqueeze(0)


def run_pipeline(
    image_path,
    dino_encoder,
    detector,
    clip_model,
    mlp_classifier,
    label_encoder,
    supercat_map,
    device,
):
    """
    Full detection + classification on a single image.

    1. DINOv2 CLS token → detector → predicted count K + K boxes
    2. For each box: crop → CLIP → MLP → class label

    Returns list of dicts: {bbox, class_name, confidence}
        bbox: [x1, y1, x2, y2] normalized to [0, 1]
    """
    pil_image    = Image.open(str(image_path)).convert("RGB")
    image_tensor = io.read_image(str(image_path))
    W, H         = pil_image.size

    # ── Detect ────────────────────────────────────────────────────────────────
    dino_encoder.eval()
    detector.eval()
    with torch.no_grad():
        cls_token    = dino_encoder(_dino_preprocess(image_tensor).to(device))  # (1, D)
        count_logits, box_preds = detector(cls_token)                            # (1, K_cls), (1, MAX_DET, 4)

    pred_count = count_logits.argmax(dim=1).item()
    pred_boxes = box_preds[0, :pred_count].cpu()  # (pred_count, 4)

    if pred_count == 0:
        return []

    # ── Classify each crop ────────────────────────────────────────────────────
    clip_model.eval()
    mlp_classifier.eval()
    detections = []

    for box in pred_boxes:
        x1, y1, x2, y2 = box.tolist()
        if (x2 - x1) < 1e-3 or (y2 - y1) < 1e-3:
            continue

        crop_t = _clip_crop(pil_image, (x1, y1, x2, y2), W, H).to(device)

        with torch.no_grad():
            emb  = clip_model.encode_image(crop_t)
            prob = torch.softmax(mlp_classifier(emb), dim=1)
            conf, idx = prob.max(dim=1)

        supercat_id = label_encoder.idx2id.get(idx.item())
        detections.append({
            "bbox":       [x1, y1, x2, y2],
            "class_name": supercat_map.get(supercat_id, f"cls_{idx.item()}"),
            "confidence": round(conf.item(), 3),
        })

    return detections
