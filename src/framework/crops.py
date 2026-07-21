"""Rare-class-aware crop utilities for segmentation training."""
from __future__ import annotations

import random

import numpy as np


def select_rare_class_crop(
    mask: np.ndarray,
    crop_size: tuple[int, int],
    rare_class_ids: list[int],
    min_rare_fraction: float = 0.01,
    attempts: int = 10,
    rng: random.Random | None = None,
) -> tuple[int, int, int, int]:
    """Return a crop box that preferentially contains rare-class pixels."""
    generator = rng or random
    height, width = mask.shape[-2:]
    crop_h, crop_w = min(crop_size[0], height), min(crop_size[1], width)
    max_top, max_left = height - crop_h, width - crop_w
    best_box = (0, 0, crop_w, crop_h)
    best_fraction = -1.0
    rare = np.isin(mask, rare_class_ids)
    for _ in range(max(1, attempts)):
        top = generator.randint(0, max_top) if max_top > 0 else 0
        left = generator.randint(0, max_left) if max_left > 0 else 0
        fraction = float(rare[top : top + crop_h, left : left + crop_w].mean())
        box = (left, top, left + crop_w, top + crop_h)
        if fraction > best_fraction:
            best_fraction, best_box = fraction, box
        if fraction >= min_rare_fraction:
            return box
    return best_box
