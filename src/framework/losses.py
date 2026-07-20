"""Loss registry for segmentation experiments."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module, weight: float = 0.5) -> None:
        super().__init__()
        self.first, self.second, self.weight = first, second, weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.first(logits, target) + (1.0 - self.weight) * self.second(logits, target)


def build_loss(config: dict[str, Any]) -> nn.Module:
    cfg = config["training"].get("loss", {"name": "cross_entropy"})
    name = str(cfg.get("name", "cross_entropy")).lower()
    if name in {"cross_entropy", "ce"}:
        weights = cfg.get("class_weights")
        tensor = torch.tensor(weights, dtype=torch.float32) if weights else None
        return nn.CrossEntropyLoss(weight=tensor)
    if name == "dice":
        return DiceLoss(float(cfg.get("smooth", 1.0)))
    if name == "focal":
        return FocalLoss(float(cfg.get("gamma", 2.0)))
    if name in {"ce_dice", "cross_entropy_dice"}:
        return CombinedLoss(nn.CrossEntropyLoss(), DiceLoss(), float(cfg.get("ce_weight", 0.5)))
    raise ValueError(f"Unsupported loss: {name}")
