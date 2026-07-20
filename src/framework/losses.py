"""Loss registry for segmentation experiments."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        probs = logits.softmax(dim=1) * valid.unsqueeze(1)
        one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1)
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        ce = ce[valid]
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
    weights = cfg.get("class_weights")
    tensor = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    ignore_index = int(cfg.get("ignore_index", config.get("data", {}).get("ignore_index", 255)))
    if name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(weight=tensor, ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(float(cfg.get("smooth", 1.0)), ignore_index)
    if name == "focal":
        return FocalLoss(float(cfg.get("gamma", 2.0)), tensor, ignore_index)
    if name in {"ce_dice", "cross_entropy_dice"}:
        return CombinedLoss(
            nn.CrossEntropyLoss(weight=tensor, ignore_index=ignore_index),
            DiceLoss(float(cfg.get("smooth", 1.0)), ignore_index),
            float(cfg.get("ce_weight", 0.5)),
        )
    raise ValueError(f"Unsupported loss: {name}")
