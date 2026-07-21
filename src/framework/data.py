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

from .crops import select_rare_class_crop
from .labels import mapping_from_config, remap_mask
from .sampling import build_weighted_sampler, class_histogram, effective_number_weights, sample_weights_from_histograms


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
        rare_crop_size: tuple[int, int] | None = None,
        rare_class_ids: list[int] | None = None,
        min_rare_fraction: float = 0.01,
        rare_crop_attempts: int = 10,
    ) -> None:
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        self.image_size = image_size
        self.label_mapping = label_mapping
        self.ignore_index = ignore_index
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.brightness_jitter = float(brightness_jitter)
        self.rare_crop_size = rare_crop_size
        self.rare_class_ids = rare_class_ids or []
        self.min_rare_fraction = float(min_rare_fraction)
        self.rare_crop_attempts = int(rare_crop_attempts)
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected {self.image_dir} and {self.mask_dir}. Use the documented split/images and split/masks layout.")
        self.images = sorted(path for path in self.image_dir.iterdir() if path.suffix.lower() in self.IMAGE_SUFFIXES)
        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")
        missing = [path.name for path in self.images if not (self.mask_dir / f"{path.stem}.png").exists()]
        if missing:
            raise FileNotFoundError(f"Missing PNG masks for: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.images)

    def load_mask_array(self, index: int) -> np.ndarray:
        image_path = self.images[index]
        mask = Image.open(self.mask_dir / f"{image_path.stem}.png")
        array = np.asarray(mask, dtype=np.int64).copy()
        if self.label_mapping is not None:
            array = remap_mask(array, self.label_mapping, self.ignore_index).astype(np.int64)
        return array

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        mask_array = self.load_mask_array(index)
        mask = Image.fromarray(mask_array.astype(np.int32), mode="I")

        if self.rare_crop_size is not None and self.rare_class_ids:
            box = select_rare_class_crop(mask_array, self.rare_crop_size, self.rare_class_ids, self.min_rare_fraction, self.rare_crop_attempts)
            image = image.crop(box)
            mask = mask.crop(box)
        if random.random() < self.horizontal_flip_probability:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if self.brightness_jitter > 0:
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter))

        height, width = self.image_size
        image = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [height, width], interpolation=TF.InterpolationMode.NEAREST)
        return TF.to_tensor(image), torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())


def build_dataset(config: dict[str, Any], split: str, training: bool = False) -> SegmentationFolderDataset:
    data_cfg = config["data"]
    augmentation_cfg = config.get("augmentation", {}) if training else {}
    crop_cfg = config.get("training", {}).get("rare_crop", {}) if training else {}
    crop_size = crop_cfg.get("size") if bool(crop_cfg.get("enabled", False)) else None
    return SegmentationFolderDataset(
        Path(data_cfg["root"]), split, tuple(int(value) for value in data_cfg["image_size"]),
        label_mapping=mapping_from_config(data_cfg.get("label_mapping")),
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        horizontal_flip_probability=float(augmentation_cfg.get("horizontal_flip_probability", 0.0)),
        brightness_jitter=float(augmentation_cfg.get("brightness_jitter", 0.0)),
        rare_crop_size=tuple(int(v) for v in crop_size) if crop_size else None,
        rare_class_ids=[int(v) for v in crop_cfg.get("class_ids", [])],
        min_rare_fraction=float(crop_cfg.get("min_fraction", 0.01)),
        rare_crop_attempts=int(crop_cfg.get("attempts", 10)),
    )


def _dataset_statistics(dataset: SegmentationFolderDataset, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    histograms = np.stack([class_histogram([dataset.load_mask_array(index)], num_classes, dataset.ignore_index) for index in range(len(dataset))])
    return histograms.sum(axis=0), histograms


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg, train_cfg = config["data"], config["training"]
    train_dataset = build_dataset(config, str(data_cfg["train_split"]), training=True)
    val_dataset = build_dataset(config, str(data_cfg["validation_split"]), training=False)
    common = {"batch_size": int(train_cfg["batch_size"]), "num_workers": int(train_cfg.get("num_workers", 0)), "pin_memory": torch.cuda.is_available()}
    sampling_cfg = train_cfg.get("sampling", {})
    strategy = str(sampling_cfg.get("strategy", "uniform")).lower()
    sampler = None
    if strategy not in {"uniform", "weighted"}:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")
    if strategy == "weighted" or bool(train_cfg.get("loss", {}).get("auto_class_weights", False)):
        counts, histograms = _dataset_statistics(train_dataset, int(data_cfg["num_classes"]))
        weights = effective_number_weights(counts, float(sampling_cfg.get("beta", 0.9999)))
        if bool(train_cfg.get("loss", {}).get("auto_class_weights", False)):
            train_cfg.setdefault("loss", {})["class_weights"] = weights.tolist()
        if strategy == "weighted":
            sampler = build_weighted_sampler(sample_weights_from_histograms(histograms, weights), int(config["experiment"].get("seed", 42)))
    return DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, **common), DataLoader(val_dataset, shuffle=False, **common)
