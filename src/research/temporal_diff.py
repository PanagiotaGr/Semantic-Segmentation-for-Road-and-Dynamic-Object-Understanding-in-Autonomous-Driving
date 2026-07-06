"""Temporal difference maps for video semantic segmentation."""

from __future__ import annotations

import torch


def temporal_change_maps(predictions: torch.Tensor) -> torch.Tensor:
    """Return binary maps showing pixel label changes between frames.

    Args:
        predictions: Tensor with shape [frames, height, width].

    Returns:
        Boolean tensor with shape [frames - 1, height, width].
    """

    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [frames, height, width]")
    if predictions.shape[0] < 2:
        raise ValueError("at least two frames are required")
    return predictions[1:] != predictions[:-1]


def temporal_change_count(predictions: torch.Tensor) -> torch.Tensor:
    """Count how many times each pixel changes label over a sequence."""

    return temporal_change_maps(predictions).sum(dim=0)


def temporal_change_frequency(predictions: torch.Tensor) -> torch.Tensor:
    """Return per-pixel change frequency in the range [0, 1]."""

    maps = temporal_change_maps(predictions)
    return maps.float().mean(dim=0)


__all__ = [
    "temporal_change_maps",
    "temporal_change_count",
    "temporal_change_frequency",
]
