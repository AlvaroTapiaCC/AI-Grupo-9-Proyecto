import torch
from pathlib import Path


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    path = Path(path)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model