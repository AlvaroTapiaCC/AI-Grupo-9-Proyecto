import torch
import torchvision.transforms as transforms
from .. import config


def get_train_transforms():

    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])


def get_val_transforms():

    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])