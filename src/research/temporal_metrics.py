"""Temporal metrics for video semantic segmentation."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _validate_sequence(predictions: torch.Tensor) -> None:
    if predictions.ndim != 3:
        raise ValueError(
            f"predictions must have shape [frames, height, width], received {tuple(predictions.shape)}"
        )
    if predictions.shape[0] < 2:
        raise ValueError("at least two frames are required for temporal metrics")


def frame_change_rate(predictions: torch.Tensor, ignore_index: Optional[int] = None) -> torch.Tensor:
    """Return the mean pixel label-change rate between consecutive frames."""

    _validate_sequence(predictions)
    previous = predictions[:-1]
    current = predictions[1:]
    changed = previous != current

    if ignore_index is not None:
        valid = (previous != ignore_index) & (current != ignore_index)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        return changed[valid].float().mean()

    return changed.float().mean()


def consecutive_frame_iou(
    predictions: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute mean IoU between consecutive predicted masks."""

    _validate_sequence(predictions)
    scores = []

    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue

        previous = predictions[:-1] == class_id
        current = predictions[1:] == class_id

        intersection = (previous & current).sum(dim=(1, 2)).float()
        union = (previous | current).sum(dim=(1, 2)).float()
        valid = union > 0

        if valid.any():
            scores.append((intersection[valid] / union[valid].clamp_min(1.0)).mean())

    if not scores:
        return torch.tensor(0.0, device=predictions.device)

    return torch.stack(scores).mean()


def temporal_stability_score(
    predictions: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Return a compact temporal stability report."""

    change_rate = frame_change_rate(predictions, ignore_index=ignore_index)
    frame_iou = consecutive_frame_iou(predictions, num_classes=num_classes, ignore_index=ignore_index)

    return {
        "frame_change_rate": change_rate,
        "consecutive_frame_iou": frame_iou,
        "stability": 1.0 - change_rate,
    }


__all__ = [
    "frame_change_rate",
    "consecutive_frame_iou",
    "temporal_stability_score",
]
