"""Uncertainty utilities for semantic segmentation.

These functions are intentionally lightweight and independent from the
training scripts. They can be used with logits from U-Net, DeepLabV3+,
SegFormer, or future segmentation models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class UncertaintyMaps:
    """Container for segmentation confidence and uncertainty outputs.

    Attributes:
        probabilities: Softmax probabilities with shape ``[B, C, H, W]``.
        prediction: Predicted class indices with shape ``[B, H, W]``.
        confidence: Maximum class probability with shape ``[B, H, W]``.
        entropy: Shannon entropy with shape ``[B, H, W]``.
        normalized_entropy: Entropy divided by log(num_classes), in ``[0, 1]``.
        risk: Simple risk score defined as ``1 - confidence``.
    """

    probabilities: torch.Tensor
    prediction: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    risk: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return the maps as a plain dictionary."""

        return {
            "probabilities": self.probabilities,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "risk": self.risk,
        }


def _validate_segmentation_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape [batch, classes, height, width], "
            f"but received shape {tuple(tensor.shape)}."
        )
    if tensor.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two classes.")


def probability_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pixel-wise Shannon entropy from class probabilities.

    Args:
        probabilities: Tensor with shape ``[B, C, H, W]``.
        eps: Numerical-stability constant.

    Returns:
        Entropy tensor with shape ``[B, H, W]``.
    """

    _validate_segmentation_tensor(probabilities, "probabilities")
    safe_probabilities = probabilities.clamp_min(eps)
    return -(safe_probabilities * safe_probabilities.log()).sum(dim=1)


def logits_to_uncertainty(logits: torch.Tensor, eps: float = 1e-8) -> UncertaintyMaps:
    """Convert segmentation logits into prediction and uncertainty maps.

    Args:
        logits: Raw model outputs with shape ``[B, C, H, W]``.
        eps: Numerical-stability constant.

    Returns:
        ``UncertaintyMaps`` containing probabilities, prediction, confidence,
        entropy, normalized entropy, and risk.
    """

    _validate_segmentation_tensor(logits, "logits")
    probabilities = F.softmax(logits, dim=1)
    return probabilities_to_uncertainty(probabilities, eps=eps)


def probabilities_to_uncertainty(
    probabilities: torch.Tensor,
    eps: float = 1e-8,
    renormalize: bool = True,
) -> UncertaintyMaps:
    """Convert probabilities into prediction and uncertainty maps.

    Args:
        probabilities: Class probabilities with shape ``[B, C, H, W]``.
        eps: Numerical-stability constant.
        renormalize: If true, re-normalize probabilities along the class axis.

    Returns:
        ``UncertaintyMaps`` containing standard uncertainty diagnostics.
    """

    _validate_segmentation_tensor(probabilities, "probabilities")

    if renormalize:
        probabilities = probabilities.clamp_min(eps)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(eps)

    confidence, prediction = probabilities.max(dim=1)
    entropy = probability_entropy(probabilities, eps=eps)
    num_classes = probabilities.shape[1]
    max_entropy = torch.log(torch.tensor(float(num_classes), device=probabilities.device))
    normalized_entropy = entropy / max_entropy.clamp_min(eps)
    risk = 1.0 - confidence

    return UncertaintyMaps(
        probabilities=probabilities,
        prediction=prediction,
        confidence=confidence,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        risk=risk,
    )


def mc_dropout_uncertainty(
    stochastic_logits: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Estimate uncertainty from multiple stochastic forward passes.

    Args:
        stochastic_logits: Tensor with shape ``[T, B, C, H, W]``, where ``T``
            is the number of stochastic forward passes.
        eps: Numerical-stability constant.

    Returns:
        Dictionary with predictive probabilities, predictive entropy, expected
        entropy, mutual information, confidence, and prediction.
    """

    if stochastic_logits.ndim != 5:
        raise ValueError(
            "stochastic_logits must have shape [passes, batch, classes, height, width], "
            f"but received shape {tuple(stochastic_logits.shape)}."
        )
    if stochastic_logits.shape[0] < 2:
        raise ValueError("At least two stochastic passes are required for MC dropout uncertainty.")

    probabilities = F.softmax(stochastic_logits, dim=2)
    predictive_probabilities = probabilities.mean(dim=0)

    predictive_entropy = probability_entropy(predictive_probabilities, eps=eps)
    expected_entropy = probability_entropy(probabilities.flatten(0, 1), eps=eps)
    expected_entropy = expected_entropy.view(stochastic_logits.shape[0], stochastic_logits.shape[1], *stochastic_logits.shape[-2:]).mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy

    confidence, prediction = predictive_probabilities.max(dim=1)

    return {
        "predictive_probabilities": predictive_probabilities,
        "prediction": prediction,
        "confidence": confidence,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "risk": 1.0 - confidence,
    }


def threshold_risk_map(risk: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Create a binary risk mask from a continuous risk map.

    Args:
        risk: Risk tensor with any shape.
        threshold: Values greater than or equal to this threshold are risky.

    Returns:
        Boolean tensor with the same shape as ``risk``.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    return risk >= threshold


__all__ = [
    "UncertaintyMaps",
    "logits_to_uncertainty",
    "probabilities_to_uncertainty",
    "probability_entropy",
    "mc_dropout_uncertainty",
    "threshold_risk_map",
]
