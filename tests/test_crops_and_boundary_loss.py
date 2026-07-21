import random

import numpy as np
import torch

from src.framework.crops import select_rare_class_crop
from src.framework.losses import BoundaryLoss


def test_rare_crop_prefers_pedestrian_region() -> None:
    mask = np.zeros((12, 12), dtype=np.int64)
    mask[8:12, 8:12] = 4
    box = select_rare_class_crop(
        mask,
        (4, 4),
        [4],
        min_rare_fraction=0.25,
        attempts=100,
        rng=random.Random(7),
    )
    left, top, right, bottom = box
    assert np.mean(mask[top:bottom, left:right] == 4) >= 0.25


def test_rare_crop_handles_images_smaller_than_crop() -> None:
    mask = np.zeros((3, 5), dtype=np.int64)
    assert select_rare_class_crop(mask, (8, 8), [4]) == (0, 0, 5, 3)


def test_boundary_loss_is_lower_for_correct_prediction() -> None:
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    target[:, 2:6, 2:6] = 1
    correct = torch.full((1, 2, 8, 8), -6.0)
    correct[:, 0] = torch.where(target == 0, 6.0, -6.0)
    correct[:, 1] = torch.where(target == 1, 6.0, -6.0)
    incorrect = torch.zeros_like(correct)
    loss = BoundaryLoss()
    assert loss(correct, target) < loss(incorrect, target)


def test_boundary_loss_ignores_void_pixels() -> None:
    target = torch.full((1, 4, 4), 255, dtype=torch.long)
    logits = torch.randn((1, 2, 4, 4), requires_grad=True)
    value = BoundaryLoss(ignore_index=255)(logits, target)
    assert torch.isfinite(value)
