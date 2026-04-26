from src.config import encode, level
from src.features import clip_encoder
from src.paths import TRAIN_EMB, TEST_EMB, VAL_EMB

from src.training.train_mlp import train_mlp
from src.training.evaluate_mlp import evaluate_mlp


def main():

    if encode:
        print("[INFO] Building embeddings...")
        clip_encoder.build_embeddings()
        print("[DONE]")
    else:
        print("[INFO] Embeddings already built")

    print("[INFO] Training MLP...")
    model, test_metrics = train_mlp(
        TRAIN_EMB,
        TEST_EMB
    )

    print("\n[INFO] Final test metrics:")
    for metric in test_metrics.keys():
        print(f"    {metric}: {test_metrics[metric]}")

    print("\n[INFO] Evaluating on val set...")
    val_metrics = evaluate_mlp(
        model,
        VAL_EMB,
        device=model.model[0].weight.device
    )

    print("[INFO] Val metrics:")
    for metric in val_metrics.keys():
        print(f"    {metric}: {val_metrics[metric]}")

    print("")
    

if __name__ == "__main__":
    main()