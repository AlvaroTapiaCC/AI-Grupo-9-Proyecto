import torch
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, path, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "metrics": metrics,
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", None)

    return model, optimizer, epoch, metrics


def save_best_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)