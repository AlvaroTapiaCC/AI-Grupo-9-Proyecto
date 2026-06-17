import torch
import torch.nn as nn

from .. import config


class RetailDetector(nn.Module):
    """
    Count + box detector on top of a frozen DINOv2 CLS token.

    Input:  (B, feature_dim)  — precomputed CLS token
    Output:
        counts  (B, num_count_classes)  — logits for 0..MAX_DET products
        boxes   (B, MAX_DET, 4)         — sigmoid-normalized [x1,y1,x2,y2] in [0,1]
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

        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.max_detections * 4),
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        counts = self.count_head(x)                                      # (B, num_count_classes)
        boxes  = torch.sigmoid(self.box_head(x))                         # (B, MAX_DET*4)
        boxes  = boxes.view(x.size(0), self.max_detections, 4)           # (B, MAX_DET, 4)
        return counts, boxes
