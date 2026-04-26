import json
from pathlib import Path


class LabelEncoder:

    def __init__(self):
        self.id2idx = {}
        self.idx2id = {}

    def fit(self, labels):
        unique_labels = sorted(set(labels))

        self.id2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2id = {idx: label for label, idx in self.id2idx.items()}

    def transform(self, labels):
        return [self.id2idx[l] for l in labels]

    def inverse_transform(self, indices):
        return [self.idx2id[i] for i in indices]

    def num_classes(self):
        return len(self.id2idx)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "id2idx": self.id2idx,
                    "idx2id": self.idx2id,
                },
                f,
                indent=4,
            )

    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(path, "r") as f:
            data = json.load(f)

        enc = cls()
        enc.id2idx = {k: v for k, v in data["id2idx"].items()}
        enc.idx2id = {int(k): v for k, v in data["idx2id"].items()}

        return enc