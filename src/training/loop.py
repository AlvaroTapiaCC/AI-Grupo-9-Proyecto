import torch
from torch.utils.data import TensorDataset


def load_embeddings(path) -> TensorDataset:
    """Load precomputed embeddings file → TensorDataset(embeddings, labels)."""
    data = torch.load(path, weights_only=True)
    return TensorDataset(data["embeddings"], data["labels"])


def load_detector_features(path) -> TensorDataset:
    """Load precomputed detector file → TensorDataset(cls_tokens, counts, boxes)."""
    data = torch.load(path, weights_only=True)
    return TensorDataset(data["cls_tokens"], data["counts"], data["boxes"])


def run_epoch_classifier(loader, model, criterion, optimizer, device):
    """Single train or eval epoch for the MLP classifier.

    Returns: (avg_loss, accuracy)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            n          += x.size(0)

    return total_loss / n, correct / n


def run_epoch_detector(loader, model, loss_fn, optimizer, device):
    """Single train or eval epoch for the count+box detector.

    Returns: (avg_count_loss, avg_box_loss, avg_count_mae)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_count_loss, total_box_loss, total_mae, n = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for cls_tokens, count_targets, box_targets in loader:
            cls_tokens    = cls_tokens.to(device)
            count_targets = count_targets.to(device)
            box_targets   = box_targets.to(device)

            count_logits, box_preds = model(cls_tokens)
            count_loss, box_loss    = loss_fn(count_logits, box_preds, count_targets, box_targets)
            loss = count_loss + box_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = cls_tokens.size(0)
            total_count_loss += count_loss.item() * B
            total_box_loss   += box_loss.item() * B
            total_mae        += (count_logits.argmax(dim=1) - count_targets).abs().float().sum().item()
            n                += B

    return total_count_loss / n, total_box_loss / n, total_mae / n
