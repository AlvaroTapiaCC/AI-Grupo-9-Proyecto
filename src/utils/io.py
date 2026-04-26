import json
from pathlib import Path
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)

    # Convert numpy / torch types safely
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=convert)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_history(history, path):
    save_json(path, history)


def save_metrics(metrics, path):
    # handle confusion matrix explicitly
    if "confusion_matrix" in metrics:
        metrics = dict(metrics)
        metrics["confusion_matrix"] = np.array(metrics["confusion_matrix"]).tolist()

    save_json(path, metrics)


def save_config(config_dict, path):
    save_json(path, config_dict)