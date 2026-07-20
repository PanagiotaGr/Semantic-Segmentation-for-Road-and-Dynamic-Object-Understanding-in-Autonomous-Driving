"""Dataset utilities for configuration-driven segmentation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


class SegmentationFolderDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load paired images and integer masks from split/images and split/masks."""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(self, root: Path, split: str, image_size: tuple[int, int]) -> None:
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        self.image_size = image_size
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
        height, width = self.image_size
        image = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [height, width], interpolation=TF.InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return image_tensor, mask_tensor


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    size = tuple(int(value) for value in data_cfg["image_size"])
    root = Path(data_cfg["root"])
    train_dataset = SegmentationFolderDataset(root, data_cfg["train_split"], size)
    val_dataset = SegmentationFolderDataset(root, data_cfg["validation_split"], size)
    common = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader
