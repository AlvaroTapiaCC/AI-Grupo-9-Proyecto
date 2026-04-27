import json
import shutil
import torch.nn as nn
import torch.optim as optim

from .. import config

from ..models.cnn import CNNClassifier
from ..data.dataset_loader import RetailDataset, get_dataloaders
from ..data.label_encoder import LabelEncoder
from ..data.transforms import get_train_transforms, get_val_transforms

from ..paths import (
    TRAIN_ANNOTATIONS,
    TEST_ANNOTATIONS,
    VAL_ANNOTATIONS,
    LABEL_ENCODER_PATH,
    LAST_MODEL_PATH,
    BEST_MODEL_PATH,
    LAST_METRICS_PATH,
    BEST_METRICS_PATH,
)

from .metrics import get_predictions, compute_all_metrics, plot_and_save_confusion_matrix
from .diagnostics import analyze_training, compare_with_best
from .training_utils import run_epoch

from ..utils.model_io import save_model, load_model
from ..utils.io import save_history, save_metrics, save_config



def train_cnn():
    device = config.device

    label_encoder = LabelEncoder.load(LABEL_ENCODER_PATH)
    num_classes = label_encoder.num_classes()

    train_dataset = RetailDataset(
        TRAIN_ANNOTATIONS,
        split="train",
        transform=get_train_transforms(),
        label_encoder_path=LABEL_ENCODER_PATH
    )

    val_dataset = RetailDataset(
        VAL_ANNOTATIONS,
        split="val",
        transform=get_val_transforms(),
        label_encoder_path=LABEL_ENCODER_PATH
    )

    test_dataset = RetailDataset(
        TEST_ANNOTATIONS,
        split="test",
        transform=get_val_transforms(),
        label_encoder_path=LABEL_ENCODER_PATH
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config.batch_size
    )

    model = CNNClassifier(
        in_channels=3,
        num_classes=num_classes,
        image_h=config.image_size[0],
        image_w=config.image_size[1]
    ).to(device)

    best_global_path = BEST_MODEL_PATH / "best.pt"
    if not config.train_new and best_global_path.exists():
        model = load_model(model, best_global_path, device)
        model.eval()

        print("[INFO] Loaded best global CNN model")

        y_true, y_pred = get_predictions(model, test_loader, device)
        metrics = compute_all_metrics(y_true, y_pred)

        save_metrics(metrics, LAST_METRICS_PATH / "metrics.json")
        plot_and_save_confusion_matrix(
            y_true, y_pred, LAST_METRICS_PATH / "confusion_matrix.png"
        )

        return model, metrics

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    best_test_acc = 0.0
    best_model_state = None
    best_epoch = 0

    LAST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    LAST_METRICS_PATH.mkdir(parents=True, exist_ok=True)
    BEST_METRICS_PATH.mkdir(parents=True, exist_ok=True)

    save_config(
        {
            "level": config.level,
            "device": config.device,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": config.lr,
            "model": config.model,
        },
        LAST_MODEL_PATH / "config.json",
    )

    for epoch in range(config.epochs):

        train_loss, train_acc = run_epoch(
            train_loader, model, criterion, optimizer, device
        )

        test_loss, test_acc = run_epoch(
            val_loader, model, criterion, None, device
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict()
            best_epoch = epoch

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {test_loss:.4f} acc {test_acc:.4f}"
        )

    model.load_state_dict(best_model_state)

    save_model(model, LAST_MODEL_PATH / "last.pt")

    y_true, y_pred = get_predictions(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_pred)

    save_history(history, LAST_MODEL_PATH / "history.json")
    save_metrics(metrics, LAST_METRICS_PATH / "metrics.json")
    plot_and_save_confusion_matrix(
        y_true, y_pred, LAST_METRICS_PATH / "confusion_matrix.png"
    )

    if config.train_new:
        status = analyze_training(
            {
                "train_acc": [history["train_acc"][best_epoch]],
                "test_acc": [history["test_acc"][best_epoch]],
                "train_loss": [history["train_loss"][best_epoch]],
                "test_loss": [history["test_loss"][best_epoch]],
            }
        )
        print(f"\n[DIAGNOSTIC] {status}")

    best_metrics_file = BEST_METRICS_PATH / "metrics.json"
    best_history_file = BEST_MODEL_PATH / "history.json"

    is_better = True
    if best_metrics_file.exists() and best_history_file.exists():
        with open(best_metrics_file, "r") as f:
            best_metrics = json.load(f)
        with open(best_history_file, "r") as f:
            best_history = json.load(f)

        rel_status = compare_with_best(metrics, best_metrics, history, best_history)
        print(f"[DIAGNOSTIC] {rel_status}\n")

        is_better = metrics["accuracy"] > best_metrics["accuracy"]

    if is_better:
        print("[INFO] New global best CNN model")

        save_metrics(metrics, BEST_METRICS_PATH / "metrics.json")
        plot_and_save_confusion_matrix(
            y_true, y_pred, BEST_METRICS_PATH / "confusion_matrix.png"
        )

        save_model(model, BEST_MODEL_PATH / "best.pt")
        shutil.copy(LAST_MODEL_PATH / "history.json", BEST_MODEL_PATH / "history.json")
        shutil.copy(LAST_MODEL_PATH / "config.json", BEST_MODEL_PATH / "config.json")

    return model, metrics