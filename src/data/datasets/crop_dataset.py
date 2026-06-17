import torch
from torch.utils.data import Dataset


class CropEmbeddingDataset(Dataset):
    """Precomputed embeddings from GT bbox crops (CLIP or DINOv2 classifier)."""

    def __init__(self, path):
        data = torch.load(path, weights_only=True)
        self.embeddings = data["embeddings"]
        self.labels     = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    @property
    def input_dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def num_samples(self) -> int:
        return len(self.labels)
