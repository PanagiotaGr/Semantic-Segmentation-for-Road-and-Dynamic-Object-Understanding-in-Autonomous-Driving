"""Loss functions for semantic segmentation experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss for semantic segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss for hard-example emphasis."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets.long(), weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedCrossEntropyDiceLoss(nn.Module):
    """Weighted combination of cross-entropy and Dice loss."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets.long())
        dice = self.dice(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice
