import torch
from PIL import Image
from torch.utils.data import TensorDataset

def run_epoch(loader, model, criterion, optimizer, device):
    total_loss = 0
    correct = 0
    total = 0

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device)
            
            logits = model(x)   
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total

def load_embeddings(path):
    data = torch.load(path)
    return TensorDataset(data["embeddings"], data["labels"])

def load_tensors(path):
    data = torch.load(path)
    return TensorDataset(data["images"], data["labels"])


def predict_bboxes(
    model,
    device,
    image_path,
    ann_data,
    image_id,
    label_encoder,
    cat_map,
    preprocess,
    clip_model
    
):
    image = Image.open(image_path).convert("RGB")

    results = []

    model.eval()
    if clip_model is not None:
        clip_model.eval()

    with torch.no_grad():
        for ann in ann_data["annotations"]:
            if ann["image_id"] != image_id:
                continue

            bbox = ann["bbox"]
            x, y, w, h = bbox

            supercat = cat_map.get(ann["category_id"])
            if supercat is None:
                continue

            true_label = label_encoder.transform([supercat])[0]

            crop = image.crop((x, y, x + w, y + h))
            crop = preprocess(crop).unsqueeze(0).to(device)

            if clip_model is not None:
                emb = clip_model.encode_image(crop)
                logits = model(emb)
            else:
                logits = model(crop)

            pred_label = logits.argmax(dim=1).item()

            results.append((bbox, true_label, pred_label))

    return image, results