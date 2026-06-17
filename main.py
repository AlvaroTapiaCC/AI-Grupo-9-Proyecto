import time
import warnings
warnings.filterwarnings("ignore")

import clip

from src import config
from src.paths import (
    CLIP_TRAIN_EMB, CLIP_LABEL_ENCODER,
    DINO_TRAIN_EMB, DINO_LABEL_ENCODER,
    CLS_BEST_CHECKPOINT, DET_BEST_CHECKPOINT,
    DET_LAST_RESULTS,
)
from src.data.label_encoder import LabelEncoder
from src.data.datasets.crop_dataset import CropEmbeddingDataset
from src.models.mlp import MLPClassifier
from src.models.detector import RetailDetector
from src.encoders.clip import build_embeddings as build_clip_embeddings
from src.encoders.dinov2 import DINOv2Encoder, build_crop_embeddings, build_detector_features
from src.stages.classifier.train import train_classifier
from src.stages.classifier.evaluate import evaluate_classifier
from src.stages.detector.train import train_detector
from src.stages.detector.evaluate import evaluate_detector, visualize_detector_predictions
from src.stages.pipeline.evaluate import evaluate_pipeline
from src.utils.model_io import load_model
from src.data.data_utils import build_supercategory_name_mapping
from src.utils.io import load_json
from src.paths import SUPERCATEGORIES_PATH


def _load_classifier(emb_path, label_encoder_path, checkpoint):
    label_encoder = LabelEncoder.load(label_encoder_path)
    dataset       = CropEmbeddingDataset(emb_path)
    model         = MLPClassifier(dataset.input_dim, label_encoder.num_classes()).to(config.device)
    return load_model(model, checkpoint, config.device), label_encoder


def _load_detector(checkpoint):
    model = RetailDetector().to(config.device)
    return load_model(model, checkpoint, config.device)


def _load_dino_encoder():
    encoder = DINOv2Encoder(model_name=config.dinov2_model, freeze=True).to(config.device)
    encoder.eval()
    return encoder


def main():
    start = time.time()
    print(f"[INFO] Mode: {config.model.upper()} | Level: {config.level} | Device: {config.device}\n")

    # ── CLASSIFIER (CLIP MLP) ─────────────────────────────────────────────────
    if config.model == "classifier":

        if config.encode:
            print("[INFO] Building CLIP embeddings...")
            build_clip_embeddings()

        if config.train_new:
            model, is_better = train_classifier()
        else:
            print("[INFO] Loading best classifier checkpoint...")
            model, _ = _load_classifier(CLIP_TRAIN_EMB, CLIP_LABEL_ENCODER, CLS_BEST_CHECKPOINT)
            is_better = False

        print("\n[INFO] Evaluating classifier on val set...")
        metrics = evaluate_classifier(model, is_better)

    # ── CLASSIFIER (DINOv2 MLP) ───────────────────────────────────────────────
    elif config.model == "dinov2":

        if config.encode:
            print("[INFO] Building DINOv2 crop embeddings...")
            build_crop_embeddings()

        if config.train_new:
            model, is_better = train_classifier(DINO_TRAIN_EMB, None, DINO_LABEL_ENCODER)
        else:
            print("[INFO] Loading best DINOv2 classifier checkpoint...")
            model, _ = _load_classifier(DINO_TRAIN_EMB, DINO_LABEL_ENCODER, CLS_BEST_CHECKPOINT)
            is_better = False

        print("\n[INFO] Evaluating DINOv2 classifier on val set...")
        metrics = evaluate_classifier(model, is_better, val_emb=None, label_encoder_path=DINO_LABEL_ENCODER)

    # ── DETECTOR ──────────────────────────────────────────────────────────────
    elif config.model == "detector":

        if config.encode:
            print("[INFO] Building detector features...")
            build_detector_features()

        dino_encoder = _load_dino_encoder()

        if config.train_new:
            model, is_better = train_detector()
        else:
            print("[INFO] Loading best detector checkpoint...")
            model    = _load_detector(DET_BEST_CHECKPOINT)
            is_better = False

        print("\n[INFO] Evaluating detector on val set...")
        metrics = evaluate_detector(model, is_better)

        from src.paths import DET_BEST_RESULTS
        vis_dir = DET_LAST_RESULTS if config.train_new else DET_BEST_RESULTS
        print("\n[INFO] Saving detector visualizations...")
        visualize_detector_predictions(model, dino_encoder, save_dir=vis_dir)

    # ── PIPELINE (detector + CLIP classifier) ─────────────────────────────────
    elif config.model == "pipeline":

        print("[INFO] Loading models...")
        dino_encoder  = _load_dino_encoder()
        detector      = _load_detector(DET_BEST_CHECKPOINT)
        clip_mlp, label_encoder = _load_classifier(CLIP_TRAIN_EMB, CLIP_LABEL_ENCODER, CLS_BEST_CHECKPOINT)
        clip_model, _ = clip.load("ViT-B/32", device=config.device)
        supercat_map  = build_supercategory_name_mapping(load_json(SUPERCATEGORIES_PATH))

        print("\n[INFO] Evaluating full pipeline on val set...")
        metrics = evaluate_pipeline(
            dino_encoder, detector, clip_model,
            clip_mlp, label_encoder, supercat_map,
            n_images=50,
        )

    else:
        raise ValueError(
            f"Unknown model: '{config.model}'. "
            "Choose 'classifier' | 'dinov2' | 'detector' | 'pipeline'."
        )

    print("\n[INFO] Final metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    elapsed = time.time() - start
    print(f"\n[INFO] Runtime: {int(elapsed // 60)}m {int(elapsed % 60)}s\n")


if __name__ == "__main__":
    main()
