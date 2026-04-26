# src/training/train_mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.mlp import MLPClassifier
from .. import config

from ..data.label_encoder import LabelEncoder
from ..paths import EMBEDDINGS_PATH

from .metrics import get_predictions, compute_all_metrics
from ..utils.checkpointing import save_checkpoint
from ..utils.io import save_history, save_metrics, save_config


def load_embeddings(path):
    data = torch.load(path)
    return TensorDataset(data["embeddings"], data["labels"])


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


def train_mlp(train_path, test_path):
    device = config.device

    train_dataset = load_embeddings(train_path)
    test_dataset = load_embeddings(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # 1. LOAD LABEL ENCODER (PERSISTENCE FIX)
    label_encoder = LabelEncoder.load(
        EMBEDDINGS_PATH / "label_encoder.json"
    )

    # 2. INPUT DIM FROM EMBEDDINGS
    input_dim = train_dataset.tensors[0].shape[1]

    # 3. NUM_CLASSES FROM ENCODER (FIX)
    num_classes = label_encoder.num_classes()

    model = MLPClassifier(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    best_test_acc = 0.0

    # 4. SAVE CONFIG FOR REPRODUCIBILITY
    save_config(
        {
        "level": config.level,
        "device": config.device,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "model": config.model,
        },
        "results/mlp/config.json"
    )

    for epoch in range(config.epochs):

        train_loss, train_acc = run_epoch(
            train_loader, model, criterion, optimizer, device
        )

        test_loss, test_acc = run_epoch(
            test_loader, model, criterion, None, device
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        metrics_snapshot = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

        # checkpoint last
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path="results/mlp/checkpoints/last.pt",
            metrics=metrics_snapshot,
        )

        # checkpoint best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path="results/mlp/checkpoints/best.pt",
                metrics=metrics_snapshot,
            )

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

    # FINAL EVALUATION
    y_true, y_pred = get_predictions(model, test_loader, device)
    val_metrics = compute_all_metrics(y_true, y_pred)

    # SAVE RESULTS
    save_history(history, "results/mlp/history.json")
    save_metrics(val_metrics, "results/mlp/metrics.json")

    return model, val_metrics