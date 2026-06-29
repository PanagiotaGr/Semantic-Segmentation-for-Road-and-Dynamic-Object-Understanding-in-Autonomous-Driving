"""Advanced evaluation metrics for semantic segmentation.

The functions in this file are model-agnostic and can be used with U-Net,
DeepLabV3+, SegFormer, PSPNet, PAN, FPN, LinkNet, and other segmentation models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationReport:
    """Container for per-class segmentation metrics."""

    table: pd.DataFrame
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_iou: float


def confusion_matrix_from_masks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Compute a dense pixel-level confusion matrix."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    true_flat = y_true.reshape(-1).astype(np.int64)
    pred_flat = y_pred.reshape(-1).astype(np.int64)

    valid = (true_flat >= 0) & (true_flat < num_classes)
    valid &= (pred_flat >= 0) & (pred_flat < num_classes)

    if ignore_index is not None:
        valid &= true_flat != ignore_index

    encoded = num_classes * true_flat[valid] + pred_flat[valid]
    matrix = np.bincount(encoded, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
) -> ClassificationReport:
    """Compute precision, recall, F1, and IoU from a confusion matrix."""
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)

    table = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support_pixels": matrix.sum(axis=1),
        }
    )

    return ClassificationReport(
        table=table,
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        mean_iou=float(np.mean(iou)),
    )


def normalized_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize confusion matrix rows to show class-wise error distribution."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums > 0)
