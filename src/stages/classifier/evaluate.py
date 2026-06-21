import shutil
from torch.utils.data import DataLoader

from ... import config
from ...paths import (
    CLIP_VAL_EMB, CLIP_LABEL_ENCODER,
    CLS_LAST_RESULTS, CLS_BEST_RESULTS,
    CLS_LAST_LOGS, CLS_BEST_LOGS,
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

    results_dir  = CLS_LAST_RESULTS
    logs_dir     = CLS_LAST_LOGS
    history_dir  = CLS_LAST_LOGS if config.train_new else CLS_BEST_LOGS

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    for f in results_dir.glob("*.png"):
        f.unlink()

    val_dataset   = CropEmbeddingDataset(val_emb)
    val_loader    = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    label_encoder = LabelEncoder.load(label_encoder_path)

    y_true, y_pred = get_classifier_predictions(model, val_loader, config.device)
    metrics        = compute_classifier_metrics(y_true, y_pred)

    save_json(logs_dir / "val_metrics.json", metrics)

    supercat_names = list(
        build_supercategory_name_mapping(load_json(SUPERCATEGORIES_PATH)).values()
    )

    from ...visualization.plots import plot_confusion_matrix, plot_classifier_history, plot_latent_space
    from ...visualization.predictions import show_classifier_predictions
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import numpy as np

    history_path = history_dir / "history.json"
    if history_path.exists():
        plot_classifier_history(load_json(history_path), results_dir)

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=supercat_names,
        save_path=results_dir / "confusion_matrix.png",
    )
    show_classifier_predictions(model, label_encoder, config.device, save_dir=results_dir)

    # ── Latent space analysis ─────────────────────────────────────────────────
    supercat_map  = build_supercategory_name_mapping(load_json(SUPERCATEGORIES_PATH))
    ordered_names = [supercat_map[label_encoder.idx2id[i]]
                     for i in range(label_encoder.num_classes())]

    embeddings = val_dataset.embeddings.numpy()
    labels_np  = val_dataset.labels.numpy()

    plot_latent_space(embeddings, labels_np, ordered_names, results_dir)

    n_cls    = label_encoder.num_classes()
    km_preds = KMeans(n_clusters=n_cls, random_state=42, n_init="auto").fit_predict(embeddings)
    ari      = adjusted_rand_score(labels_np, km_preds)
    nmi      = normalized_mutual_info_score(labels_np, km_preds)
    print(f"[INFO] Latent space clustering (K-Means, K={n_cls}):  ARI={ari:.4f}  NMI={nmi:.4f}")

    if config.train_new and is_better:
        if CLS_BEST_RESULTS.exists():
            shutil.rmtree(CLS_BEST_RESULTS)
        shutil.copytree(CLS_LAST_RESULTS, CLS_BEST_RESULTS)
        print("[INFO] Best classifier results updated.")

    return metrics
