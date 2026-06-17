import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    CLIP_TRAIN_EMB, CLIP_TEST_EMB, CLIP_LABEL_ENCODER,
    CLS_LAST_CHECKPOINT, CLS_BEST_CHECKPOINT,
    CLS_LAST_RESULTS, CLS_BEST_RESULTS,
    CLS_LAST_LOGS, CLS_BEST_LOGS,
)
from ...models.mlp import MLPClassifier
from ...data.label_encoder import LabelEncoder
from ...data.datasets.crop_dataset import CropEmbeddingDataset
from ...training.loop import run_epoch_classifier
from ...training.metrics import get_classifier_predictions, compute_classifier_metrics
from ...utils.model_io import save_model
from ...utils.io import save_json, load_json


def train_classifier(train_emb=None, test_emb=None, label_encoder_path=None):
    train_emb          = train_emb          or CLIP_TRAIN_EMB
    test_emb           = test_emb           or CLIP_TEST_EMB
    label_encoder_path = label_encoder_path or CLIP_LABEL_ENCODER

    train_dataset = CropEmbeddingDataset(train_emb)
    test_dataset  = CropEmbeddingDataset(test_emb)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)

    label_encoder = LabelEncoder.load(label_encoder_path)
    model         = MLPClassifier(train_dataset.input_dim, label_encoder.num_classes()).to(config.device)
    optimizer     = optim.Adam(model.parameters(), lr=config.lr)
    criterion     = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_test_acc, best_state, best_epoch = 0.0, None, 0

    for d in (CLS_LAST_CHECKPOINT.parent, CLS_BEST_CHECKPOINT.parent,
              CLS_LAST_RESULTS, CLS_BEST_RESULTS,
              CLS_LAST_LOGS, CLS_BEST_LOGS):
        d.mkdir(parents=True, exist_ok=True)

    col = config.epochs
    print(f"\n[INFO] Training MLPClassifier...")
    print(f"{'Epoch':>{len(str(col))+6}} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print(f"{'-'*(len(str(col))+6)}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")

    for epoch in range(config.epochs):
        train_loss, train_acc = run_epoch_classifier(train_loader, model, criterion, optimizer, config.device)
        test_loss,  test_acc  = run_epoch_classifier(test_loader,  model, criterion, None,      config.device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch

        marker = " *" if test_acc == best_test_acc else ""
        epoch_str = f"{epoch+1}/{col}"
        print(f"{epoch_str:>{len(str(col))+6}} | {train_loss:>10.4f} | {train_acc:>9.4f} | {test_loss:>8.4f} | {test_acc:>7.4f}{marker}")

    model.load_state_dict(best_state)
    save_model(model, CLS_LAST_CHECKPOINT)

    y_true, y_pred = get_classifier_predictions(model, test_loader, config.device)
    metrics        = compute_classifier_metrics(y_true, y_pred)

    save_json(CLS_LAST_LOGS / "history.json", history)
    save_json(CLS_LAST_LOGS / "metrics.json", metrics)

    best_metrics_file = CLS_BEST_LOGS / "metrics.json"
    is_better = True
    if best_metrics_file.exists():
        is_better = metrics["accuracy"] > load_json(best_metrics_file)["accuracy"]

    if is_better:
        print("[INFO] New best classifier — saving...")
        save_model(model, CLS_BEST_CHECKPOINT)
        if CLS_BEST_LOGS.exists():
            shutil.rmtree(CLS_BEST_LOGS)
        shutil.copytree(CLS_LAST_LOGS, CLS_BEST_LOGS)

    return model, is_better
