"""Visualization utilities for semantic segmentation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # Background
    1: (128, 64, 128),  # Road
    2: (244, 35, 232),  # Sidewalk
    3: (0, 0, 142),     # Vehicle
    4: (220, 20, 60),   # Pedestrian
}


def colorize_mask(
    mask: np.ndarray,
    color_map: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Convert a class-index mask into an RGB mask."""
    colors = color_map or DEFAULT_CLASS_COLORS
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        rgb[mask == class_id] = color

    return rgb


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color_map: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Overlay a semantic mask on an RGB image."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be an RGB array with shape H x W x 3")
    if image.shape[:2] != mask.shape:
        raise ValueError("image and mask must have matching spatial dimensions")

    mask_rgb = colorize_mask(mask, color_map)
    image_float = image.astype(np.float32)
    mask_float = mask_rgb.astype(np.float32)
    blended = (1.0 - alpha) * image_float + alpha * mask_float
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_comparison_figure(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    title: str = "Semantic segmentation comparison",
) -> None:
    """Save a three-panel comparison: image, ground truth, prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (image, "Input image"),
        (colorize_mask(ground_truth), "Ground truth"),
        (colorize_mask(prediction), "Prediction"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)

    for axis, (panel, panel_title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(panel_title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an RGB image as a NumPy array."""
    return np.asarray(Image.open(path).convert("RGB"))
