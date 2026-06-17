import shutil
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    CLIP_VAL_EMB, CLIP_LABEL_ENCODER,
    CLS_LAST_RESULTS, CLS_BEST_RESULTS,
    SUPERCATEGORIES_PATH,
)
from ...data.label_encoder import LabelEncoder
from ...data.datasets.crop_dataset import CropEmbeddingDataset
from ...data.data_utils import build_supercategory_name_mapping
from ...training.metrics import get_classifier_predictions, compute_classifier_metrics
from ...utils.io import save_json, load_json


def evaluate_classifier(model, is_better, val_emb=None, label_encoder_path=None):
    val_emb            = val_emb            or CLIP_VAL_EMB
    label_encoder_path = label_encoder_path or CLIP_LABEL_ENCODER

    val_dataset   = CropEmbeddingDataset(val_emb)
    val_loader    = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    label_encoder = LabelEncoder.load(label_encoder_path)

    y_true, y_pred = get_classifier_predictions(model, val_loader, config.device)
    metrics        = compute_classifier_metrics(y_true, y_pred)

    CLS_LAST_RESULTS.mkdir(parents=True, exist_ok=True)
    save_json(CLS_LAST_RESULTS / "val_metrics.json", metrics)

    supercat_names = list(
        build_supercategory_name_mapping(load_json(SUPERCATEGORIES_PATH)).values()
    )

    from ...visualization.plots import plot_confusion_matrix
    from ...visualization.predictions import show_classifier_predictions

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=supercat_names,
        save_path=CLS_LAST_RESULTS / "confusion_matrix.png",
    )
    show_classifier_predictions(model, label_encoder, config.device, save_dir=CLS_LAST_RESULTS)

    if is_better:
        CLS_BEST_RESULTS.mkdir(parents=True, exist_ok=True)
        shutil.copy(CLS_LAST_RESULTS / "val_metrics.json",     CLS_BEST_RESULTS / "val_metrics.json")
        shutil.copy(CLS_LAST_RESULTS / "confusion_matrix.png", CLS_BEST_RESULTS / "confusion_matrix.png")

    return metrics
