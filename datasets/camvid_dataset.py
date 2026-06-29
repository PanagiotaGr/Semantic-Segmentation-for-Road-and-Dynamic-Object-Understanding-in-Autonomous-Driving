"""Reusable CamVid dataset utilities for segmentation experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


CLASS_COLORS = {
    (128, 128, 128): 0,
    (128, 0, 0): 1,
    (192, 192, 128): 2,
    (128, 64, 128): 3,
    (0, 0, 192): 4,
    (128, 128, 0): 5,
    (192, 128, 128): 6,
    (64, 64, 128): 7,
    (64, 0, 128): 8,
    (64, 64, 0): 9,
    (0, 128, 192): 10,
    (0, 0, 0): 11,
}

REDUCED_MAP = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 0,
    6: 0,
    7: 0,
    8: 3,
    9: 4,
    10: 0,
    11: 0,
}


def rgb_to_original_mask(mask: Image.Image) -> np.ndarray:
    """Convert an RGB CamVid label image to integer class indices."""
    mask_array = np.asarray(mask)
    class_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
    for rgb, class_id in CLASS_COLORS.items():
        class_mask[np.all(mask_array == np.array(rgb), axis=-1)] = class_id
    return class_mask


def rgb_to_reduced_mask(mask: Image.Image) -> np.ndarray:
    """Convert an RGB CamVid label image to the reduced 5-class label space."""
    original = rgb_to_original_mask(mask)
    reduced = np.zeros_like(original, dtype=np.uint8)
    for original_id, reduced_id in REDUCED_MAP.items():
        reduced[original == original_id] = reduced_id
    return reduced


class CamVidSegmentationDataset(Dataset):
    """CamVid dataset wrapper returning image and segmentation mask pairs."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, image_size: int = 256, reduced: bool = True, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.reduced = reduced
        self.transform = transform
        self.image_files = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        self.mask_files = sorted([p for p in self.mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Image and mask directories must contain the same number of files")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        image = Image.open(self.image_files[index]).convert("RGB").resize((self.image_size, self.image_size))
        mask_image = Image.open(self.mask_files[index]).convert("RGB").resize((self.image_size, self.image_size), Image.NEAREST)
        mask = rgb_to_reduced_mask(mask_image) if self.reduced else rgb_to_original_mask(mask_image)
        image_array = np.asarray(image)

        if self.transform is not None:
            augmented = self.transform(image=image_array, mask=mask)
            return augmented["image"], augmented["mask"].long()

        return image_array, mask
