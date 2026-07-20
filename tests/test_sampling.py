import numpy as np

from src.framework.sampling import (
    class_histogram,
    effective_number_weights,
    sample_weights_from_histograms,
)


def test_class_histogram_ignores_unknown_labels() -> None:
    mask = np.array([[0, 1, 255], [1, 2, 7]])
    counts = class_histogram([mask], num_classes=3, ignore_index=255)
    assert counts.tolist() == [1, 2, 1]


def test_effective_number_upweights_rare_classes() -> None:
    weights = effective_number_weights(np.array([1000, 100, 10]), beta=0.99)
    assert weights[2] > weights[1] > weights[0]
    assert np.isclose(weights.sum(), 3.0)


def test_sample_weights_prefer_rare_class_samples() -> None:
    histograms = np.array([[100, 0], [50, 10]])
    weights = sample_weights_from_histograms(histograms, np.array([1.0, 5.0]))
    assert weights[1] > weights[0]
