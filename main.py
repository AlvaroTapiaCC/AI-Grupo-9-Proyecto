import time

from src import config

from src.paths import (
    TRAIN_EMB, VAL_EMB,
    TRAIN_TENS, VAL_TENS,
    LABEL_ENCODER_PATH,
    BEST_MODEL_PATH,
    BEST_MLP_HISTORY, BEST_CNN_HISTORY,
    COMPARISON_PATH
)

from src.features import clip_encoder
from src.features import tensor_builder

from src.models.mlp import MLPClassifier
from src.models.cnn import CNNClassifier

from src.data.label_encoder import LabelEncoder
from src.training.training_utils import load_embeddings, load_tensors

from src.training.train_mlp import train_mlp
from src.training.evaluate_mlp import evaluate_mlp

from src.training.train_cnn import train_cnn
from src.training.evaluate_cnn import evaluate_cnn

from src.utils.model_io import load_model

from src.results.plots import plot_model_comparison


def main():

    start = time.time()
    
    print(f"[INFO] Selected model: {config.model.upper()}")

    # =========================
    # MLP
    # =========================
    if config.model == "mlp":

        if config.encode:
            print("[INFO] Building embeddings...")
            clip_encoder.build_embeddings()
            print("[DONE]")
        else:
            print("[INFO] Embeddings already built")

        if config.train_new:
            model_obj, is_better = train_mlp()
            
            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_mlp(
                model_obj,
                is_better,
                VAL_EMB,
                device=config.device,
                batch_size=config.batch_size
            )

        else:
            print("[INFO] Using Pre Trained MLP...")

            label_encoder = LabelEncoder.load(LABEL_ENCODER_PATH)
            train_dataset = load_embeddings(TRAIN_EMB)

            input_dim = train_dataset.tensors[0].shape[1]
            num_classes = label_encoder.num_classes()

            model_obj = MLPClassifier(input_dim, num_classes).to(config.device)
            model_obj = load_model(
                model_obj,
                BEST_MODEL_PATH / "best.pt",
                config.device
            )

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_mlp(
                model_obj,
                is_better=False,
                val_path=VAL_EMB,
                device=config.device,
                batch_size=config.batch_size
            )

    # =========================
    # CNN
    # =========================
    elif config.model == "cnn":

        if config.encode:
            print("[INFO] Building tensors...")
            tensor_builder.build_all_tensors()
            print("[DONE]")
        else:
            print("[INFO] Tensors already built")

        if config.train_new:
            model_obj, is_better = train_cnn()
            
            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_cnn(
                model_obj,
                is_better,
                VAL_TENS,
                device=config.device,
                batch_size=config.batch_size
            )

        else:
            print("[INFO] Using Pre Trained CNN...")

            label_encoder = LabelEncoder.load(LABEL_ENCODER_PATH)
            train_dataset = load_tensors(TRAIN_TENS)

            _, C, H, W = train_dataset.tensors[0].shape
            num_classes = label_encoder.num_classes()

            model_obj = CNNClassifier(
                in_channels=C,
                num_classes=num_classes,
                image_h=H,
                image_w=W
            ).to(config.device)
            model_obj = load_model(
                model_obj,
                BEST_MODEL_PATH / "best.pt",
                config.device
            )

            print("\n[INFO] Evaluating on val set...")
            val_metrics = evaluate_cnn(
                model_obj,
                is_better=False,
                val_path=VAL_TENS,
                device=config.device,
                batch_size=config.batch_size
            )

    else:
        raise ValueError(f"Unknown model type: {config.model}")

    print("\n[INFO] Val metrics:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v}")
        
    if config.compare and (BEST_MLP_HISTORY is not None and BEST_CNN_HISTORY is not None):
        print("[INFO] comparing best MLP with best CNN")
        plot_model_comparison(BEST_MLP_HISTORY, BEST_CNN_HISTORY, COMPARISON_PATH)
    
    end = time.time()
    total_time = end - start
    minutes = total_time // 60
    seconds = total_time - (minutes * 60)
    print(f"\n[INFO] Runtime: {minutes:.0f} minutes and {seconds:.0f} seconds\n")


if __name__ == "__main__":
    main()