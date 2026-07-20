"""Label-space utilities for CamVid experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np

CAMVID_12_TO_5 = {
    0: 0,   # sky -> background
    1: 0,   # building -> background
    2: 0,   # pole -> background
    3: 1,   # road -> road
    4: 2,   # pavement -> sidewalk
    5: 0,   # tree -> background
    6: 0,   # sign symbol -> background
    7: 0,   # fence -> background
    8: 3,   # car -> vehicle
    9: 4,   # pedestrian -> pedestrian
    10: 3,  # bicyclist -> vehicle/dynamic object
    11: 0,  # unlabeled -> background
}


def build_lookup(mapping: dict[int, int], ignore_index: int = 255) -> np.ndarray:
    """Build a dense uint8 lookup table for fast mask remapping."""
    table = np.full(256, ignore_index, dtype=np.uint8)
    for source, target in mapping.items():
        if not 0 <= int(source) <= 255:
            raise ValueError(f"Source label out of range: {source}")
        table[int(source)] = int(target)
    return table


def remap_mask(mask: np.ndarray, mapping: dict[int, int], ignore_index: int = 255) -> np.ndarray:
    """Remap an integer mask while preserving unknown labels as ignore_index."""
    if mask.dtype.kind not in {"u", "i"}:
        raise TypeError("Segmentation masks must contain integer class ids")
    if mask.min(initial=0) < 0 or mask.max(initial=0) > 255:
        raise ValueError("Mask labels must be in the range 0..255")
    return build_lookup(mapping, ignore_index)[mask.astype(np.uint8)]


def mapping_from_config(values: Iterable[int] | dict[int, int] | None) -> dict[int, int] | None:
    """Parse either a dense list or explicit dictionary from YAML."""
    if values is None:
        return None
    if isinstance(values, dict):
        return {int(key): int(value) for key, value in values.items()}
    return {index: int(value) for index, value in enumerate(values)}
