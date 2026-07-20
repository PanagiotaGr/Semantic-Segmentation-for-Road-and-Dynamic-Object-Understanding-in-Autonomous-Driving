"""Metrics for continual semantic segmentation experiments."""

from __future__ import annotations

from typing import Dict

import torch


def _validate_performance_matrix(scores: torch.Tensor) -> None:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [training_step, evaluation_domain]")
    if scores.shape[0] < 2:
        raise ValueError("at least two training steps are required")
    if scores.shape[1] < 1:
        raise ValueError("at least one evaluation domain is required")


def average_accuracy(scores: torch.Tensor) -> torch.Tensor:
    """Return average performance after the final continual-learning step."""

    _validate_performance_matrix(scores)
    return scores[-1].mean()


def forgetting_score(scores: torch.Tensor) -> torch.Tensor:
    """Compute mean forgetting over domains seen before the final step.

    For each previous domain, forgetting is the best historical score minus
    the final score. Higher values indicate more forgetting.
    """

    _validate_performance_matrix(scores)
    if scores.shape[0] == 2:
        historical_best = scores[:-1].max(dim=0).values
    else:
        historical_best = scores[:-1].max(dim=0).values
    forgetting = historical_best - scores[-1]
    return forgetting.clamp_min(0).mean()


def retained_performance(scores: torch.Tensor) -> torch.Tensor:
    """Return final performance divided by best historical performance."""

    _validate_performance_matrix(scores)
    best = scores.max(dim=0).values.clamp_min(1e-8)
    return (scores[-1] / best).mean()


def forward_transfer(scores: torch.Tensor) -> torch.Tensor:
    """Estimate forward transfer using upper-triangular future-domain scores."""

    _validate_performance_matrix(scores)
    values = []
    steps, domains = scores.shape
    for step in range(steps):
        for domain in range(step + 1, domains):
            values.append(scores[step, domain])
    if not values:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(values).mean()


def continual_learning_report(scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return a compact continual-learning metric report."""

    return {
        "average_accuracy": average_accuracy(scores),
        "forgetting_score": forgetting_score(scores),
        "retained_performance": retained_performance(scores),
        "forward_transfer": forward_transfer(scores),
    }


__all__ = [
    "average_accuracy",
    "forgetting_score",
    "retained_performance",
    "forward_transfer",
    "continual_learning_report",
]
