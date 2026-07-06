"""Simple temporal feature-fusion baseline modules."""

from __future__ import annotations

import torch
from torch import nn


class TemporalFeatureFusion(nn.Module):
    """Fuse per-frame feature maps using mean, sum, or concatenation.

    Expected input shape:
        [batch, frames, channels, height, width]
    """

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "sum", "concat"}:
            raise ValueError("mode must be 'mean', 'sum', or 'concat'")
        self.mode = mode

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape [batch, frames, channels, height, width]")
        if self.mode == "mean":
            return features.mean(dim=1)
        if self.mode == "sum":
            return features.sum(dim=1)
        batch, frames, channels, height, width = features.shape
        return features.reshape(batch, frames * channels, height, width)


class TemporalFusionSegmentationHead(nn.Module):
    """Minimal fusion head for temporal segmentation experiments."""

    def __init__(self, input_channels: int, num_classes: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        if fused_features.ndim != 4:
            raise ValueError("fused_features must have shape [batch, channels, height, width]")
        return self.head(fused_features)


__all__ = ["TemporalFeatureFusion", "TemporalFusionSegmentationHead"]
