"""Reporting helpers for publication-oriented segmentation evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_confusion_matrix_plot(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: Path,
    normalize: bool = True,
) -> None:
    """Save a readable confusion-matrix figure."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if len(class_names) != values.shape[0]:
        raise ValueError("Class names must match matrix dimensions")
    if normalize:
        denominator = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values)
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    threshold = values.max(initial=0) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = f"{values[row, column]:.2f}" if normalize else str(int(values[row, column]))
            axis.text(column, row, text, ha="center", va="center", color="white" if values[row, column] > threshold else "black")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def concatenate_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect predictions and labels for research-only post-hoc metrics."""
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels.cpu())
    if not predictions:
        raise ValueError("Evaluation loader is empty")
    return torch.cat(predictions), torch.cat(targets)
