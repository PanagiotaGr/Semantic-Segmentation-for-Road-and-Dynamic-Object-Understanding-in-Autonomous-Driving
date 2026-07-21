"""Loss registry for segmentation experiments."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth, self.ignore_index = smooth, ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        probs = logits.softmax(dim=1) * valid.unsqueeze(1)
        one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, ignore_index: int = 255) -> None:
        super().__init__()
        self.gamma, self.ignore_index = gamma, ignore_index
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index, reduction="none")
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        ce = ce[valid]
        return (((1.0 - torch.exp(-ce)) ** self.gamma) * ce).mean()


def _soft_boundaries(values: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    dilation = F.max_pool2d(values, kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-values, kernel_size, stride=1, padding=padding)
    return (dilation - erosion).clamp(0.0, 1.0)


class BoundaryLoss(nn.Module):
    """Dice-style loss on class boundaries extracted by morphological gradients."""

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0, kernel_size: int = 3) -> None:
        super().__init__()
        self.ignore_index, self.smooth, self.kernel_size = ignore_index, smooth, kernel_size

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        probabilities = logits.softmax(dim=1) * valid.unsqueeze(1)
        labels = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        predicted_boundary = _soft_boundaries(probabilities, self.kernel_size)
        target_boundary = _soft_boundaries(labels, self.kernel_size)
        dims = (0, 2, 3)
        intersection = (predicted_boundary * target_boundary).sum(dims)
        denominator = predicted_boundary.sum(dims) + target_boundary.sum(dims)
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module, weight: float = 0.5) -> None:
        super().__init__()
        self.first, self.second, self.weight = first, second, weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.first(logits, target) + (1.0 - self.weight) * self.second(logits, target)


def build_loss(config: dict[str, Any]) -> nn.Module:
    cfg = config["training"].get("loss", {"name": "cross_entropy"})
    name = str(cfg.get("name", "cross_entropy")).lower()
    weights = cfg.get("class_weights")
    tensor = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    ignore_index = int(cfg.get("ignore_index", config.get("data", {}).get("ignore_index", 255)))
    ce = nn.CrossEntropyLoss(weight=tensor, ignore_index=ignore_index)
    dice = DiceLoss(float(cfg.get("smooth", 1.0)), ignore_index)
    boundary = BoundaryLoss(ignore_index, float(cfg.get("boundary_smooth", 1.0)), int(cfg.get("boundary_kernel_size", 3)))
    if name in {"cross_entropy", "ce"}:
        return ce
    if name == "dice":
        return dice
    if name == "focal":
        return FocalLoss(float(cfg.get("gamma", 2.0)), tensor, ignore_index)
    if name == "boundary":
        return boundary
    if name in {"ce_dice", "cross_entropy_dice"}:
        return CombinedLoss(ce, dice, float(cfg.get("ce_weight", 0.5)))
    if name in {"ce_boundary", "cross_entropy_boundary"}:
        return CombinedLoss(ce, boundary, float(cfg.get("ce_weight", 0.8)))
    if name in {"ce_dice_boundary", "cross_entropy_dice_boundary"}:
        region = CombinedLoss(ce, dice, float(cfg.get("ce_weight", 0.5)))
        return CombinedLoss(region, boundary, float(cfg.get("region_weight", 0.8)))
    raise ValueError(f"Unsupported loss: {name}")
