"""Publication-oriented segmentation metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _binary_boundary(mask: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    """Extract a binary boundary map from BxHxW class masks."""
    if mask.ndim != 3:
        raise ValueError("Expected masks with shape BxHxW")
    mask = mask.float().unsqueeze(1)
    kernel = 2 * dilation + 1
    maximum = F.max_pool2d(mask, kernel, stride=1, padding=dilation)
    minimum = -F.max_pool2d(-mask, kernel, stride=1, padding=dilation)
    return (maximum != minimum).squeeze(1)


def boundary_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    dilation: int = 1,
) -> float:
    """Compute IoU between predicted and target semantic boundaries."""
    valid = target != ignore_index
    pred_boundary = _binary_boundary(prediction, dilation) & valid
    target_boundary = _binary_boundary(target, dilation) & valid
    intersection = (pred_boundary & target_boundary).sum().double()
    union = (pred_boundary | target_boundary).sum().double()
    return float((intersection / union).item()) if union > 0 else 1.0


def class_binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Return safety-oriented binary metrics for one semantic class."""
    valid = target != ignore_index
    predicted = (prediction == class_id) & valid
    actual = (target == class_id) & valid
    tp = (predicted & actual).sum().double()
    fp = (predicted & ~actual & valid).sum().double()
    fn = (~predicted & actual & valid).sum().double()
    tn = (~predicted & ~actual & valid).sum().double()

    def safe(numerator: torch.Tensor, denominator: torch.Tensor) -> float | None:
        return float((numerator / denominator).item()) if denominator > 0 else None

    return {
        "class_id": int(class_id),
        "true_positive_pixels": int(tp.item()),
        "false_positive_pixels": int(fp.item()),
        "false_negative_pixels": int(fn.item()),
        "true_negative_pixels": int(tn.item()),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "iou": safe(tp, tp + fp + fn),
        "miss_rate": safe(fn, tp + fn),
    }
