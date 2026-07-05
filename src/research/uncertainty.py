"""Uncertainty utilities for semantic segmentation.

The functions in this module are model-agnostic. They accept segmentation
logits or probabilities from any network that outputs tensors with shape
``[batch, classes, height, width]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class UncertaintyMaps:
    """Standard uncertainty outputs for dense semantic segmentation."""

    probabilities: torch.Tensor
    prediction: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    risk: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Return maps as a plain dictionary."""

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
            f"but received {tuple(tensor.shape)}."
        )
    if tensor.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two classes.")


def probability_entropy(probabilities: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pixel-wise Shannon entropy from class probabilities."""

    _validate_segmentation_tensor(probabilities, "probabilities")
    safe_probabilities = probabilities.clamp_min(eps)
    return -(safe_probabilities * safe_probabilities.log()).sum(dim=1)


def logits_to_uncertainty(logits: torch.Tensor, eps: float = 1e-8) -> UncertaintyMaps:
    """Convert raw segmentation logits into uncertainty maps."""

    _validate_segmentation_tensor(logits, "logits")
    probabilities = F.softmax(logits, dim=1)
    return probabilities_to_uncertainty(probabilities, eps=eps, renormalize=False)


def probabilities_to_uncertainty(
    probabilities: torch.Tensor,
    eps: float = 1e-8,
    renormalize: bool = True,
) -> UncertaintyMaps:
    """Convert class probabilities into uncertainty maps."""

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


def mc_dropout_uncertainty(stochastic_logits: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Estimate epistemic uncertainty from multiple stochastic forward passes.

    Args:
        stochastic_logits: Tensor with shape ``[passes, batch, classes, height, width]``.
        eps: Numerical-stability constant.
    """

    if stochastic_logits.ndim != 5:
        raise ValueError(
            "stochastic_logits must have shape [passes, batch, classes, height, width], "
            f"but received {tuple(stochastic_logits.shape)}."
        )
    if stochastic_logits.shape[0] < 2:
        raise ValueError("At least two stochastic passes are required.")

    probabilities = F.softmax(stochastic_logits, dim=2)
    predictive_probabilities = probabilities.mean(dim=0)

    predictive_entropy = probability_entropy(predictive_probabilities, eps=eps)
    flat_probabilities = probabilities.flatten(0, 1)
    expected_entropy = probability_entropy(flat_probabilities, eps=eps)
    expected_entropy = expected_entropy.view(
        stochastic_logits.shape[0],
        stochastic_logits.shape[1],
        *stochastic_logits.shape[-2:],
    ).mean(dim=0)
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
    """Create a binary high-risk mask from a continuous risk map."""

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
