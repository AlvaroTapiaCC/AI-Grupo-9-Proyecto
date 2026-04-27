import time

from src.paths import TRAIN_EMB, TEST_EMB, VAL_EMB

from src import config 
from src.features import clip_encoder

from src.training.train_mlp import train_mlp
from src.training.evaluate_mlp import evaluate_mlp

from src.training.train_cnn import train_cnn
from src.training.evaluate_cnn import evaluate_cnn


def main():

    start = time.time()

    if config.encode:
        print("[INFO] Building embeddings...")
        clip_encoder.build_embeddings()
        print("[DONE]")
    else:
        print("[INFO] Embeddings already built (skipping)")

    print(f"[INFO] Selected model: {config.model.upper()}")

    if config.model == "mlp":

        if not config.train_new:
            print("[INFO] Using Pre Trained MLP...")
            model_obj, test_metrics = train_mlp(
                TRAIN_EMB,
                TEST_EMB
            )
        else:
            model_obj, test_metrics = train_mlp(
                TRAIN_EMB,
                TEST_EMB
            )

        print("\n[INFO] Final test metrics:")
        for k, v in test_metrics.items():
            print(f"    {k}: {v}")

        print("\n[INFO] Evaluating on val set...")
        val_metrics = evaluate_mlp(
            model_obj,
            VAL_EMB,
            device=model_obj.model[0].weight.device,
            batch_size=config.batch_size
        )

    elif config.model == "cnn":

        if not config.train_new:
            print("[INFO] Using Pre Trained CNN...")
            model_obj, test_metrics = train_cnn()
        else:
            model_obj, test_metrics = train_cnn()

        print("\n[INFO] Final test metrics:")
        for k, v in test_metrics.items():
            print(f"    {k}: {v}")

        print("\n[INFO] Evaluating on val set...")
        val_metrics = evaluate_cnn(
            model_obj,
            device=model_obj.device if hasattr(model_obj, "device") else None
        )

    else:
        raise ValueError(f"Unknown model type: {config.model}")

    print("[INFO] Val metrics:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v}")

    end = time.time()

    print(f"\n[INFO] Runtime: {end-start:.4f} seconds\n")


if __name__ == "__main__":
    main()