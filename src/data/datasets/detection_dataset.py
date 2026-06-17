import torch
from torch.utils.data import Dataset


class DetectionFeatureDataset(Dataset):
    """Precomputed DINOv2 CLS + patch tokens, GT counts and sorted boxes (detector)."""

    def __init__(self, path):
        data = torch.load(path, weights_only=True)
        self.cls_tokens   = data["cls_tokens"]    # (N, D)
        self.patch_tokens = data["patch_tokens"]  # (N, N_patches, D)
        self.counts       = data["counts"]        # (N,)
        self.boxes        = data["boxes"]         # (N, MAX_DET, 4)

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        return self.cls_tokens[idx], self.patch_tokens[idx], self.counts[idx], self.boxes[idx]

    @property
    def feature_dim(self) -> int:
        return self.cls_tokens.shape[1]

    @property
    def num_patches(self) -> int:
        return self.patch_tokens.shape[1]

    @property
    def max_detections(self) -> int:
        return self.boxes.shape[1]

    @property
    def num_samples(self) -> int:
        return len(self.counts)
