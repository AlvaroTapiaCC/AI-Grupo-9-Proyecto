import torch
import torch.nn as nn

from .. import config


class _BoxHead(nn.Module):
    """
    DETR-style cross-attention box head.
    MAX_DET learned queries each attend to the DINOv2 patch tokens to predict one box.
    """

    def __init__(self, feature_dim: int, max_detections: int, num_heads: int = 8):
        super().__init__()
        self.query_embed = nn.Embedding(max_detections, feature_dim)
        self.cross_attn  = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: (B, N_patches, D)
        Returns:      (B, MAX_DET, 4) sigmoid-normalized boxes
        """
        B = patch_tokens.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, MAX_DET, D)
        out, _  = self.cross_attn(queries, patch_tokens, patch_tokens)     # (B, MAX_DET, D)
        out     = self.norm(out)
        return torch.sigmoid(self.mlp(out))                                # (B, MAX_DET, 4)


class RetailDetector(nn.Module):
    """
    Count + box detector on top of frozen DINOv2 features.

    count_head : CLS token   (B, D)         → (B, num_count_classes)
    box_head   : patch tokens (B, N, D)     → (B, MAX_DET, 4)
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_count_classes: int = None,
        max_detections: int = None,
    ):
        super().__init__()
        self.max_detections = max_detections or config.max_detections
        num_count_classes   = num_count_classes or config.num_count_classes

        self.count_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_count_classes),
        )

        self.box_head = _BoxHead(feature_dim, self.max_detections)

    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor):
        """
        cls_token:    (B, D)
        patch_tokens: (B, N_patches, D)
        Returns:
            counts (B, num_count_classes)
            boxes  (B, MAX_DET, 4)
        """
        counts = self.count_head(cls_token.float())
        boxes  = self.box_head(patch_tokens.float())
        return counts, boxes
