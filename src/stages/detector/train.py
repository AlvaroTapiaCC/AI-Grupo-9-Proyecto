import shutil
import torch.optim as optim
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    DETECTOR_TRAIN_FEAT, DETECTOR_TEST_FEAT,
    DET_LAST_CHECKPOINT, DET_BEST_CHECKPOINT,
    DET_LAST_RESULTS, DET_BEST_RESULTS,
    DET_LAST_LOGS, DET_BEST_LOGS,
)
from ...models.detector import RetailDetector
from ...data.datasets.detection_dataset import DetectionFeatureDataset
from ...training.loop import run_epoch_detector
from ...training.losses import detector_loss
from ...utils.model_io import save_model
from ...utils.io import save_json, load_json


def train_detector():
    train_dataset = DetectionFeatureDataset(DETECTOR_TRAIN_FEAT)
    test_dataset  = DetectionFeatureDataset(DETECTOR_TEST_FEAT)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)

    model     = RetailDetector(feature_dim=train_dataset.feature_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    history = {
        "train_count_loss": [], "test_count_loss": [],
        "train_box_loss":   [], "test_box_loss":   [],
        "train_count_mae":  [], "test_count_mae":  [],
    }
    best_mae, best_state = float("inf"), None

    for d in (DET_LAST_CHECKPOINT.parent, DET_BEST_CHECKPOINT.parent,
              DET_LAST_RESULTS, DET_BEST_RESULTS,
              DET_LAST_LOGS, DET_BEST_LOGS):
        d.mkdir(parents=True, exist_ok=True)

    col = config.epochs
    print(f"\n[INFO] Training RetailDetector...")
    print(f"{'Epoch':>{len(str(col))+6}} | {'Tr Count':>8} | {'Val Count':>9} | {'Tr Box':>6} | {'Val Box':>7} | {'Tr MAE':>6} | {'Val MAE':>7}")
    print(f"{'-'*(len(str(col))+6)}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")

    for epoch in range(config.epochs):
        tr_cls, tr_box, tr_mae = run_epoch_detector(train_loader, model, detector_loss, optimizer, config.device)
        te_cls, te_box, te_mae = run_epoch_detector(test_loader,  model, detector_loss, None,      config.device)

        history["train_count_loss"].append(tr_cls)
        history["test_count_loss"].append(te_cls)
        history["train_box_loss"].append(tr_box)
        history["test_box_loss"].append(te_box)
        history["train_count_mae"].append(tr_mae)
        history["test_count_mae"].append(te_mae)

        if te_mae < best_mae:
            best_mae   = te_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        marker = " *" if te_mae == best_mae else ""
        epoch_str = f"{epoch+1}/{col}"
        print(f"{epoch_str:>{len(str(col))+6}} | {tr_cls:>8.4f} | {te_cls:>9.4f} | {tr_box:>6.4f} | {te_box:>7.4f} | {tr_mae:>6.4f} | {te_mae:>7.4f}{marker}")

    model.load_state_dict(best_state)
    save_model(model, DET_LAST_CHECKPOINT)
    save_json(DET_LAST_LOGS / "history.json", history)

    best_metrics_file = DET_BEST_LOGS / "metrics.json"
    is_better = True
    if best_metrics_file.exists():
        is_better = best_mae < load_json(best_metrics_file).get("count_mae", float("inf"))

    if is_better:
        print("[INFO] New best detector — saving...")
        save_model(model, DET_BEST_CHECKPOINT)
        shutil.copy(DET_LAST_LOGS / "history.json", DET_BEST_LOGS / "history.json")

    return model, is_better
