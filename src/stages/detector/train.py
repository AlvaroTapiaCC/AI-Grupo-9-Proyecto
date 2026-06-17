import shutil
import torch.optim as optim
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    DETECTOR_TRAIN_FEAT, DETECTOR_TEST_FEAT,
    DET_LAST_MODEL, DET_BEST_MODEL,
    DET_LAST_METRICS, DET_BEST_METRICS,
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
    best_mae, best_state, best_epoch = float("inf"), None, 0

    for d in (DET_LAST_MODEL, DET_BEST_MODEL, DET_LAST_METRICS, DET_BEST_METRICS):
        d.mkdir(parents=True, exist_ok=True)

    print("[INFO] Training RetailDetector...")
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
            best_epoch = epoch

        print(
            f"Epoch {epoch+1:>2}/{config.epochs} | "
            f"count {tr_cls:.4f}/{te_cls:.4f} | "
            f"box {tr_box:.4f}/{te_box:.4f} | "
            f"mae {tr_mae:.4f}/{te_mae:.4f}"
        )

    model.load_state_dict(best_state)
    save_model(model, DET_LAST_MODEL / "model.pt")
    save_json(DET_LAST_MODEL / "history.json", history)

    best_metrics_file = DET_BEST_METRICS / "metrics.json"
    is_better = True
    if best_metrics_file.exists():
        best_metrics = load_json(best_metrics_file)
        is_better    = best_mae < best_metrics.get("count_mae", float("inf"))

    if is_better:
        print("[INFO] New best detector — saving...")
        save_model(model, DET_BEST_MODEL / "model.pt")
        shutil.copy(DET_LAST_MODEL / "history.json", DET_BEST_MODEL / "history.json")

    return model, is_better
