"""Class-frequency analysis and imbalance-aware sampling utilities."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_histogram(masks: Iterable[np.ndarray], num_classes: int, ignore_index: int = 255) -> np.ndarray:
    """Aggregate pixel counts for valid classes across integer masks."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for mask in masks:
        array = np.asarray(mask, dtype=np.int64)
        valid = (array >= 0) & (array < num_classes) & (array != ignore_index)
        counts += np.bincount(array[valid], minlength=num_classes)[:num_classes]
    return counts


def effective_number_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """Return normalized class weights using the effective-number formulation."""
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.zeros_like(counts)
    positive = counts > 0
    weights[positive] = (1.0 - beta) / (1.0 - np.power(beta, counts[positive]))
    if positive.any():
        weights[positive] *= positive.sum() / weights[positive].sum()
    return weights.astype(np.float32)


def sample_weights_from_histograms(histograms: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """Score samples by the rare-class pixels they contain."""
    histograms = np.asarray(histograms, dtype=np.float64)
    class_weights = np.asarray(class_weights, dtype=np.float64)
    pixels = histograms.sum(axis=1)
    weighted = histograms @ class_weights
    return np.divide(weighted, pixels, out=np.ones_like(weighted), where=pixels > 0)


def build_weighted_sampler(sample_weights: np.ndarray, seed: int = 42) -> WeightedRandomSampler:
    """Create a deterministic replacement sampler for one training epoch."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
