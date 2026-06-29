"""Albumentations pipelines for semantic segmentation."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(image_size: int = 256, strong: bool = False) -> A.Compose:
    """Return training augmentations for image-mask pairs."""
    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
    ]

    if strong:
        transforms.extend(
            [
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.3),
            ]
        )

    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def get_val_augmentations(image_size: int = 256) -> A.Compose:
    """Return deterministic validation/test transforms."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
