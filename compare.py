from src.training.metrics import plot_model_comparison

from src.paths import BEST_MLP_HISTORY, BEST_CNN_HISTORY ,COMPARISON_PATH

def main():
    plot_model_comparison(BEST_MLP_HISTORY, BEST_CNN_HISTORY, COMPARISON_PATH)


if __name__ == "__main__":
    main()