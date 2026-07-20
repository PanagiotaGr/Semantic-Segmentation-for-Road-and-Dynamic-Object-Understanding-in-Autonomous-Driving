"""Dataset utilities for configuration-driven segmentation experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .labels import mapping_from_config, remap_mask


class SegmentationFolderDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load paired images and integer masks from split/images and split/masks."""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: tuple[int, int],
        *,
        label_mapping: dict[int, int] | None = None,
        ignore_index: int = 255,
        horizontal_flip_probability: float = 0.0,
        brightness_jitter: float = 0.0,
    ) -> None:
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        self.image_size = image_size
        self.label_mapping = label_mapping
        self.ignore_index = ignore_index
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.brightness_jitter = float(brightness_jitter)
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.mask_dir}. "
                "Use the documented split/images and split/masks layout."
            )
        self.images = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in self.IMAGE_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")
        missing = [path.name for path in self.images if not (self.mask_dir / f"{path.stem}.png").exists()]
        if missing:
            raise FileNotFoundError(f"Missing PNG masks for: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if random.random() < self.horizontal_flip_probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.brightness_jitter > 0:
            factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            image = TF.adjust_brightness(image, factor)

        height, width = self.image_size
        image = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [height, width], interpolation=TF.InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        mask_array = np.asarray(mask, dtype=np.int64).copy()
        if self.label_mapping is not None:
            mask_array = remap_mask(mask_array, self.label_mapping, self.ignore_index).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor


def build_dataset(config: dict[str, Any], split: str, training: bool = False) -> SegmentationFolderDataset:
    data_cfg = config["data"]
    augmentation_cfg = config.get("augmentation", {}) if training else {}
    return SegmentationFolderDataset(
        Path(data_cfg["root"]),
        split,
        tuple(int(value) for value in data_cfg["image_size"]),
        label_mapping=mapping_from_config(data_cfg.get("label_mapping")),
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        horizontal_flip_probability=float(augmentation_cfg.get("horizontal_flip_probability", 0.0)),
        brightness_jitter=float(augmentation_cfg.get("brightness_jitter", 0.0)),
    )


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    train_dataset = build_dataset(config, str(data_cfg["train_split"]), training=True)
    val_dataset = build_dataset(config, str(data_cfg["validation_split"]), training=False)
    common = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader
