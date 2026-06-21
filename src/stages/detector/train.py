import shutil
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    DETECTOR_TRAIN_FEAT, DETECTOR_TEST_FEAT,
    TRAIN_ANNOTATIONS, TEST_ANNOTATIONS,
    TRAIN_IMAGES, TEST_IMAGES,
    DET_LAST_CHECKPOINT, DET_BEST_CHECKPOINT,
    DINO_DET_LAST_CHECKPOINT, DINO_DET_BEST_CHECKPOINT,
    DET_LAST_RESULTS, DET_BEST_RESULTS,
    DET_LAST_LOGS, DET_BEST_LOGS,
)
from ...models.detector import RetailDetector
from ...encoders.dinov2 import DINOv2Encoder
from ...data.datasets.detection_dataset import DetectionFeatureDataset
from ...data.datasets.detection_image_dataset import DetectionImageDataset
from ...training.loop import run_epoch_detector, run_epoch_detector_finetune
from ...training.losses import detector_loss
from ...utils.model_io import save_model, load_model
from ...utils.io import save_json, load_json


def _make_dirs():
    for d in (DET_LAST_CHECKPOINT.parent, DET_BEST_CHECKPOINT.parent,
              DET_LAST_RESULTS, DET_BEST_RESULTS,
              DET_LAST_LOGS, DET_BEST_LOGS):
        d.mkdir(parents=True, exist_ok=True)


def _print_header(col):
    print(f"\n[INFO] Training RetailDetector{'  (DINOv2 fine-tune)' if config.finetune_dino else ''}...")
    print(f"[INFO] Start: {time.strftime('%H:%M:%S')}")
    print(f"{'Epoch':>{len(str(col))+6}} | {'Tr Count':>8} | {'Val Count':>9} | {'Tr Box':>6} | {'Val Box':>7} | {'Tr MAE':>6} | {'Val MAE':>7} | {'Time':>8}")
    print(f"{'-'*(len(str(col))+6)}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}")


def train_detector():
    _make_dirs()

    if config.finetune_dino:
        return _train_finetune()
    else:
        return _train_precomputed()


# ── Precomputed features (fast, frozen encoder) ───────────────────────────────

def _train_precomputed():
    train_dataset = DetectionFeatureDataset(DETECTOR_TRAIN_FEAT)
    test_dataset  = DetectionFeatureDataset(DETECTOR_TEST_FEAT)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)

    model     = RetailDetector(feature_dim=train_dataset.feature_dim).to(config.device)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer     = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    warmup_epochs = max(1, config.epochs // 10)
    scheduler     = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - warmup_epochs), eta_min=1e-5),
    ], milestones=[warmup_epochs])

    history   = _empty_history()
    best_mae, best_state = float("inf"), None
    col = config.epochs
    _print_header(col)

    for epoch in range(col):
        tr_cls, tr_box, tr_mae = run_epoch_detector(train_loader, model, detector_loss, optimizer, config.device)
        te_cls, te_box, te_mae = run_epoch_detector(test_loader,  model, detector_loss, None,      config.device)
        _record(history, tr_cls, te_cls, tr_box, te_box, tr_mae, te_mae)
        scheduler.step()

        if te_mae < best_mae:
            best_mae   = te_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        _print_row(epoch, col, tr_cls, te_cls, tr_box, te_box, tr_mae, te_mae, te_mae == best_mae)

    model.load_state_dict(best_state)
    return _save_and_return(model, None, history, best_mae)


# ── Fine-tuning (unfrozen DINOv2 blocks) ──────────────────────────────────────

def _train_finetune():
    train_dataset = DetectionImageDataset(TRAIN_ANNOTATIONS, TRAIN_IMAGES)
    test_dataset  = DetectionImageDataset(TEST_ANNOTATIONS,  TEST_IMAGES)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size,
                               shuffle=True,  num_workers=0)
    test_loader   = DataLoader(test_dataset,  batch_size=config.batch_size,
                               shuffle=False, num_workers=0)

    encoder = DINOv2Encoder(
        model_name=config.dinov2_model,
        freeze=True,
        unfreeze_blocks=config.unfreeze_blocks,
    ).to(config.device)
    model = RetailDetector(feature_dim=encoder.feature_dim).to(config.device)

    # load previous detector checkpoint as starting point if available
    if DET_BEST_CHECKPOINT.exists():
        print("[INFO] Loading previous detector weights as starting point...")
        load_model(model, DET_BEST_CHECKPOINT, config.device)

    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizer = optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": config.backbone_lr},
        {"params": model.parameters(), "lr": config.lr},
    ], weight_decay=1e-4)
    warmup_epochs = max(1, config.epochs // 10)
    scheduler     = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - warmup_epochs), eta_min=1e-6),
    ], milestones=[warmup_epochs])
    scaler = GradScaler() if config.device == "cuda" else None

    history  = _empty_history()
    best_mae, best_det_state, best_enc_state = float("inf"), None, None
    col = config.epochs
    _print_header(col)

    for epoch in range(col):
        tr_cls, tr_box, tr_mae = run_epoch_detector_finetune(
            train_loader, encoder, model, detector_loss, optimizer, config.device, scaler)
        te_cls, te_box, te_mae = run_epoch_detector_finetune(
            test_loader,  encoder, model, detector_loss, None,      config.device, None)
        _record(history, tr_cls, te_cls, tr_box, te_box, tr_mae, te_mae)
        scheduler.step()

        if te_mae < best_mae:
            best_mae      = te_mae
            best_det_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_enc_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        _print_row(epoch, col, tr_cls, te_cls, tr_box, te_box, tr_mae, te_mae, te_mae == best_mae)

    model.load_state_dict(best_det_state)
    encoder.load_state_dict(best_enc_state)
    return _save_and_return(model, encoder, history, best_mae)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_history():
    return {
        "train_count_loss": [], "test_count_loss": [],
        "train_box_loss":   [], "test_box_loss":   [],
        "train_count_mae":  [], "test_count_mae":  [],
    }


def _record(h, tr_cls, te_cls, tr_box, te_box, tr_mae, te_mae):
    h["train_count_loss"].append(tr_cls); h["test_count_loss"].append(te_cls)
    h["train_box_loss"].append(tr_box);   h["test_box_loss"].append(te_box)
    h["train_count_mae"].append(tr_mae);  h["test_count_mae"].append(te_mae)


def _print_row(epoch, col, tr_cls, te_cls, tr_box, te_box, tr_mae, te_mae, is_best):
    marker    = " *" if is_best else ""
    epoch_str = f"{epoch+1}/{col}"
    ts        = time.strftime("%H:%M:%S")
    print(f"{epoch_str:>{len(str(col))+6}} | {tr_cls:>8.4f} | {te_cls:>9.4f} | "
          f"{tr_box:>6.4f} | {te_box:>7.4f} | {tr_mae:>6.4f} | {te_mae:>7.4f} | {ts:>8}{marker}")


def _save_and_return(model, encoder, history, best_mae):
    save_model(model, DET_LAST_CHECKPOINT)
    if encoder is not None:
        save_model(encoder, DINO_DET_LAST_CHECKPOINT)
    save_json(DET_LAST_LOGS / "history.json", history)

    best_metrics_file = DET_BEST_LOGS / "metrics.json"
    is_better = True
    if best_metrics_file.exists():
        is_better = best_mae < load_json(best_metrics_file).get("count_mae", float("inf"))

    if is_better:
        print("[INFO] New best detector — saving...")
        save_model(model, DET_BEST_CHECKPOINT)
        if encoder is not None:
            save_model(encoder, DINO_DET_BEST_CHECKPOINT)
        shutil.copy(DET_LAST_LOGS / "history.json", DET_BEST_LOGS / "history.json")

    return model, encoder, is_better
