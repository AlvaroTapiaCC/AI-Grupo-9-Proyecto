import time
import warnings
warnings.filterwarnings("ignore")

from src import config

from src.paths import (
    BEST_MODEL_PATH,
    TRAIN_EMB, VAL_EMB, TEST_EMB, LABEL_ENCODER_PATH,
    TRAIN_DINO_EMB, VAL_DINO_EMB, TEST_DINO_EMB,
    DINO_LABEL_ENCODER_PATH,
    LOCATOR_FEATURES_PATH,
    TRAIN_LOC_FEAT, VAL_LOC_FEAT,
    BEST_MLP_MODEL, BEST_LOCATOR_MODEL,
    PIPELINE_RESULTS_PATH, VAL_IMAGES, SUPERCATEGORIES_PATH,
)

from src.models.mlp import MLPClassifier
from src.models.locator import RetailLocator

from src.data.label_encoder import LabelEncoder
from src.training.training_utils import load_embeddings

from src.features import clip_encoder
from src.training.evaluate_mlp import evaluate_mlp

from src.features.dinov2_crop_encoder import build_dinov2_embeddings
from src.features.dinov2_encoder import DINOv2Encoder
from src.features.locator_builder import build_locator_features

from src.training.train_mlp import train_mlp
from src.training.evaluate_dinov2 import evaluate_dinov2

from src.training.train_locator import train_locator
from src.training.evaluate_locator import evaluate_locator

from src.inference.pipeline import run_pipeline
from src.results.plots import draw_pipeline_result
from src.data.data_utils import build_supercategory_name_mapping
from src.utils.io import load_json
from src.utils.model_io import load_model


def main():

    start = time.time()
    print(f"[INFO] Selected model: {config.model.upper()}")

    # =========================
    # MLP + CLIP
    # =========================
    if config.model == "mlp":

        if config.encode:
            print("[INFO] Building CLIP embeddings...")
            clip_encoder.build_embeddings()
            print("[DONE]")
        else:
            print("[INFO] CLIP embeddings already built")

        if config.train_new:
            model_obj, is_better = train_mlp()

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_mlp(
                model_obj, is_better, VAL_EMB,
                device=config.device, batch_size=config.batch_size,
            )

        else:
            print("[INFO] Using Pre Trained CLIP MLP...")

            label_encoder = LabelEncoder.load(LABEL_ENCODER_PATH)
            train_dataset = load_embeddings(TRAIN_EMB)
            input_dim     = train_dataset.tensors[0].shape[1]
            num_classes   = label_encoder.num_classes()

            model_obj = MLPClassifier(input_dim, num_classes).to(config.device)
            model_obj = load_model(model_obj, BEST_MODEL_PATH / "best.pt", config.device)

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_mlp(
                model_obj, is_better=False, val_path=VAL_EMB,
                device=config.device, batch_size=config.batch_size,
            )

    # =========================
    # DINOV2 (CLS-token MLP classifier — uses GT bboxes)
    # =========================
    elif config.model == "dinov2":

        if config.encode:
            print("[INFO] Building DINOv2 crop embeddings...")
            build_dinov2_embeddings()
            print("[DONE]")
        else:
            print("[INFO] DINOv2 embeddings already built")

        if config.train_new:
            model_obj, is_better = train_mlp(TRAIN_DINO_EMB, TEST_DINO_EMB, DINO_LABEL_ENCODER_PATH)

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_dinov2(
                model_obj, is_better, VAL_DINO_EMB,
                device=config.device, batch_size=config.batch_size,
            )

        else:
            print("[INFO] Using Pre Trained DINOv2 MLP...")

            label_encoder = LabelEncoder.load(DINO_LABEL_ENCODER_PATH)
            input_dim     = DINOv2Encoder.MODEL_DIMS[config.dinov2_model]
            num_classes   = label_encoder.num_classes()

            model_obj = MLPClassifier(input_dim, num_classes).to(config.device)
            model_obj = load_model(model_obj, BEST_MODEL_PATH / "best.pt", config.device)

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_dinov2(
                model_obj, is_better=False, val_path=VAL_DINO_EMB,
                device=config.device, batch_size=config.batch_size,
            )

    # =========================
    # LOCATOR (binary bbox detector — class-agnostic)
    # =========================
    elif config.model == "locator":

        if config.encode:
            print("[INFO] Building locator features...")
            build_locator_features()
            print("[DONE]")
        else:
            print("[INFO] Locator features already built")

        if config.train_new:
            model_obj, is_better = train_locator()

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_locator(
                model_obj, is_better,
                device=config.device, batch_size=config.batch_size,
            )

        else:
            print("[INFO] Using Pre Trained Locator...")

            model_obj = RetailLocator(
                model_name=config.dinov2_model,
                freeze_backbone=config.freeze_backbone,
            ).to(config.device)
            model_obj = load_model(model_obj, BEST_MODEL_PATH / "best.pt", config.device)

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_locator(
                model_obj, is_better=False,
                device=config.device, batch_size=config.batch_size,
            )

    # =========================
    # PIPELINE (locator + CLIP MLP classifier)
    # =========================
    elif config.model == "pipeline":
        import random

        print("[INFO] Loading locator and CLIP MLP classifier...")

        # Locator
        locator = RetailLocator(
            model_name=config.dinov2_model,
            freeze_backbone=config.freeze_backbone,
        ).to(config.device)
        locator = load_model(locator, BEST_LOCATOR_MODEL, config.device)

        # CLIP MLP classifier
        label_encoder = LabelEncoder.load(LABEL_ENCODER_PATH)
        train_dataset = load_embeddings(TRAIN_EMB)
        input_dim     = train_dataset.tensors[0].shape[1]
        clip_mlp      = MLPClassifier(input_dim, label_encoder.num_classes()).to(config.device)
        clip_mlp      = load_model(clip_mlp, BEST_MLP_MODEL, config.device)

        # CLIP encoder
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=config.device)

        # Supercategory names
        supercat_map = build_supercategory_name_mapping(load_json(SUPERCATEGORIES_PATH))

        # Sample val images
        val_image_files = sorted(VAL_IMAGES.iterdir())
        samples = random.sample(val_image_files, min(8, len(val_image_files)))

        PIPELINE_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Running pipeline on {len(samples)} val images...")

        for i, image_path in enumerate(samples):
            detections = run_pipeline(
                image_path, locator, clip_model, clip_preprocess,
                clip_mlp, label_encoder, supercat_map, config.device,
            )
            save_path = PIPELINE_RESULTS_PATH / f"result_{i:02d}_{image_path.stem}.png"
            draw_pipeline_result(image_path, detections, save_path)
            print(f"  [{i+1}/{len(samples)}] {image_path.name} → {len(detections)} detections")

        val_metrics = {"images_processed": len(samples)}
        print(f"[INFO] Results saved to {PIPELINE_RESULTS_PATH}")

    else:
        raise ValueError(f"Unknown model type: {config.model}. Choose 'mlp', 'dinov2', 'locator', or 'pipeline'.")

    print("\n[INFO] Val metrics:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v}")

    end = time.time()
    total_time = end - start
    minutes = total_time // 60
    seconds = total_time - (minutes * 60)
    print(f"\n[INFO] Runtime: {minutes:.0f} minutes and {seconds:.0f} seconds\n")


if __name__ == "__main__":
    main()
