"""Visualization helpers for semantic-segmentation predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

DEFAULT_PALETTE = [
    (70, 70, 70),
    (128, 64, 128),
    (244, 35, 232),
    (0, 0, 142),
    (220, 20, 60),
]


def colorize_mask(mask: np.ndarray, palette: Sequence[tuple[int, int, int]] = DEFAULT_PALETTE) -> Image.Image:
    """Convert a class-id mask to an RGB image."""
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        output[mask == class_id] = color
    return Image.fromarray(output, mode="RGB")


def tensor_to_image(image: torch.Tensor) -> Image.Image:
    """Convert a normalized CHW tensor in [0, 1] to RGB PIL."""
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def save_prediction_overlay(
    image: torch.Tensor,
    prediction: torch.Tensor,
    path: Path,
    *,
    alpha: float = 0.5,
    palette: Sequence[tuple[int, int, int]] = DEFAULT_PALETTE,
) -> None:
    """Save a side-by-side RGB image, color prediction, and overlay."""
    rgb = tensor_to_image(image)
    colored = colorize_mask(prediction.detach().cpu().numpy().astype(np.int64), palette)
    overlay = Image.blend(rgb, colored, alpha)
    canvas = Image.new("RGB", (rgb.width * 3, rgb.height))
    canvas.paste(rgb, (0, 0))
    canvas.paste(colored, (rgb.width, 0))
    canvas.paste(overlay, (rgb.width * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
