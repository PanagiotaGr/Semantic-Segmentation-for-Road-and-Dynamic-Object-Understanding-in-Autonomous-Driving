"""Evaluation metrics for semantic segmentation.

This module provides reusable NumPy-based utilities for computing confusion
matrices, per-class Intersection over Union, and mean IoU. The functions are
framework-agnostic and can be used with predictions exported from PyTorch,
TensorFlow, or any other model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SegmentationMetrics:
    """Container for semantic segmentation evaluation results."""

    mean_iou: float
    class_iou: dict[str, float]
    confusion_matrix: np.ndarray


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Compute a pixel-level confusion matrix.

    Args:
        y_true: Ground-truth label mask.
        y_pred: Predicted label mask.
        num_classes: Number of semantic classes.
        ignore_index: Optional label value to ignore.

    Returns:
        A matrix where rows correspond to ground-truth classes and columns to
        predicted classes.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    true_flat = y_true.reshape(-1).astype(np.int64)
    pred_flat = y_pred.reshape(-1).astype(np.int64)

    valid = (true_flat >= 0) & (true_flat < num_classes)
    valid &= (pred_flat >= 0) & (pred_flat < num_classes)

    if ignore_index is not None:
        valid &= true_flat != ignore_index

    encoded = num_classes * true_flat[valid] + pred_flat[valid]
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def per_class_iou(confusion_matrix: np.ndarray) -> np.ndarray:
    """Compute IoU for each class from a confusion matrix."""
    true_positive = np.diag(confusion_matrix).astype(np.float64)
    false_positive = confusion_matrix.sum(axis=0) - true_positive
    false_negative = confusion_matrix.sum(axis=1) - true_positive
    denominator = true_positive + false_positive + false_negative

    return np.divide(
        true_positive,
        denominator,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def mean_iou(class_iou: Iterable[float]) -> float:
    """Compute mean IoU while ignoring NaN values."""
    values = np.asarray(list(class_iou), dtype=np.float64)
    return float(np.nanmean(values))


def evaluate_segmentation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    ignore_index: int | None = None,
) -> SegmentationMetrics:
    """Evaluate semantic segmentation predictions.

    Args:
        y_true: Ground-truth label mask.
        y_pred: Predicted label mask.
        class_names: Names of semantic classes in index order.
        ignore_index: Optional label value to ignore.

    Returns:
        SegmentationMetrics containing mIoU, class-wise IoU, and confusion
        matrix.
    """
    matrix = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=len(class_names),
        ignore_index=ignore_index,
    )
    ious = per_class_iou(matrix)
    return SegmentationMetrics(
        mean_iou=mean_iou(ious),
        class_iou={name: float(value) for name, value in zip(class_names, ious)},
        confusion_matrix=matrix,
    )
