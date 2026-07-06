"""Temporal consistency losses for video semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_logits_sequence(logits: torch.Tensor) -> None:
    if logits.ndim != 5:
        raise ValueError(
            "logits must have shape [batch, frames, classes, height, width], "
            f"received {tuple(logits.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("at least two frames are required")
    if logits.shape[2] < 2:
        raise ValueError("at least two classes are required")


def temporal_probability_consistency_loss(logits: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Penalize probability changes between consecutive frames.

    Args:
        logits: Tensor with shape [batch, frames, classes, height, width].
        reduction: Either "mean", "sum", or "none".

    Returns:
        Loss tensor. With mean or sum reduction, returns a scalar.
    """

    _validate_logits_sequence(logits)
    probabilities = F.softmax(logits, dim=2)
    differences = probabilities[:, 1:] - probabilities[:, :-1]
    loss = differences.pow(2).mean(dim=2)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def temporal_logit_consistency_loss(logits: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Penalize raw-logit changes between consecutive frames."""

    _validate_logits_sequence(logits)
    differences = logits[:, 1:] - logits[:, :-1]
    loss = differences.pow(2).mean(dim=2)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def combined_segmentation_temporal_loss(
    segmentation_loss: torch.Tensor,
    logits: torch.Tensor,
    temporal_weight: float = 0.1,
) -> torch.Tensor:
    """Combine an existing segmentation loss with temporal consistency."""

    if temporal_weight < 0:
        raise ValueError("temporal_weight must be non-negative")
    temporal_loss = temporal_probability_consistency_loss(logits)
    return segmentation_loss + temporal_weight * temporal_loss


__all__ = [
    "temporal_probability_consistency_loss",
    "temporal_logit_consistency_loss",
    "combined_segmentation_temporal_loss",
]
