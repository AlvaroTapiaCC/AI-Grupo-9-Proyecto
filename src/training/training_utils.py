import torch
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