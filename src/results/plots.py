import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import confusion_matrix

from ..utils.io import load_json

def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    save_path,
    class_names=None
):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else False,
        yticklabels=class_names if class_names is not None else False
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    #plt.show()

    return cm

def plot_and_save_training_history(history, save_path):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss while training")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path / "loss_plot.png")

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy while trining")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path / "acc_plot.png")
    #plt.show()
    
def plot_model_comparison(history_mlp_path, history_cnn_path, save_path):

    history_mlp = load_json(history_mlp_path)
    history_cnn = load_json(history_cnn_path)

    epochs_mlp = range(1, len(history_mlp["train_loss"]) + 1)
    epochs_cnn = range(1, len(history_cnn["train_loss"]) + 1)

    # --- LOSS ---
    plt.figure()
    plt.plot(epochs_mlp, history_mlp["train_loss"], color="blue", linestyle="-", label="MLP Train Loss")
    plt.plot(epochs_mlp, history_mlp["test_loss"], color="blue", linestyle="--", label="MLP Val Loss")

    plt.plot(epochs_cnn, history_cnn["train_loss"], color="red", linestyle="-", label="CNN Train Loss")
    plt.plot(epochs_cnn, history_cnn["test_loss"], color="red", linestyle="--", label="CNN Val Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Comparison (MLP vs CNN)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path / "loss_comparison.png")
    #plt.show()

    # --- ACCURACY ---
    plt.figure()
    plt.plot(epochs_mlp, history_mlp["train_acc"], color="blue", linestyle="-", label="MLP Train Acc")
    plt.plot(epochs_mlp, history_mlp["test_acc"], color="blue", linestyle="--", label="MLP Val Acc")

    plt.plot(epochs_cnn, history_cnn["train_acc"], color="red", linestyle="-", label="CNN Train Acc")
    plt.plot(epochs_cnn, history_cnn["test_acc"], color="red", linestyle="--", label="CNN Val Acc")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison (MLP vs CNN)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path / "acc_comparison.png")
    #plt.show()
        

def draw_bboxes(image, results, label_encoder, supercat_map, metrics_path):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for bbox, true_label, pred_label in results:
        x, y, w, h = bbox

        color = "green" if true_label == pred_label else "red"

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        true_name = supercat_map[label_encoder.idx2id[true_label]]
        pred_name = supercat_map[label_encoder.idx2id[pred_label]]

        ax.text(
            x, y,
            f"P: {pred_name} | T: {true_name}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6)
        )

    plt.axis("off")
    plt.savefig(metrics_path)
    #plt.show()