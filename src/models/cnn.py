import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_h: int, image_w: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        flat_features = self._get_flat_features(in_channels, image_h, image_w)

        self.classifier = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _get_flat_features(self, in_channels, h, w):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, h, w)
            x = self.features(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = x.float()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x