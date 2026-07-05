import torch

from src.research.uncertainty import (
    logits_to_uncertainty,
    mc_dropout_uncertainty,
    probability_entropy,
    probabilities_to_uncertainty,
    threshold_risk_map,
)


def test_logits_to_uncertainty_shapes():
    logits = torch.randn(2, 5, 16, 16)

    maps = logits_to_uncertainty(logits)

    assert maps.probabilities.shape == (2, 5, 16, 16)
    assert maps.prediction.shape == (2, 16, 16)
    assert maps.confidence.shape == (2, 16, 16)
    assert maps.entropy.shape == (2, 16, 16)
    assert maps.normalized_entropy.shape == (2, 16, 16)
    assert maps.risk.shape == (2, 16, 16)


def test_probabilities_are_normalized():
    logits = torch.randn(1, 5, 8, 8)
    maps = logits_to_uncertainty(logits)

    class_sums = maps.probabilities.sum(dim=1)

    assert torch.allclose(class_sums, torch.ones_like(class_sums), atol=1e-5)


def test_confidence_and_risk_are_complements():
    probabilities = torch.zeros(1, 3, 4, 4)
    probabilities[:, 1] = 0.8
    probabilities[:, 2] = 0.2

    maps = probabilities_to_uncertainty(probabilities)

    assert torch.allclose(maps.confidence, torch.full((1, 4, 4), 0.8))
    assert torch.allclose(maps.risk, torch.full((1, 4, 4), 0.2))


def test_entropy_is_higher_for_uniform_distribution():
    confident = torch.zeros(1, 4, 2, 2)
    confident[:, 0] = 1.0

    uniform = torch.full((1, 4, 2, 2), 0.25)

    confident_entropy = probability_entropy(confident).mean()
    uniform_entropy = probability_entropy(uniform).mean()

    assert uniform_entropy > confident_entropy


def test_mc_dropout_uncertainty_shapes():
    stochastic_logits = torch.randn(4, 2, 5, 8, 8)

    maps = mc_dropout_uncertainty(stochastic_logits)

    assert maps["predictive_probabilities"].shape == (2, 5, 8, 8)
    assert maps["prediction"].shape == (2, 8, 8)
    assert maps["confidence"].shape == (2, 8, 8)
    assert maps["predictive_entropy"].shape == (2, 8, 8)
    assert maps["expected_entropy"].shape == (2, 8, 8)
    assert maps["mutual_information"].shape == (2, 8, 8)
    assert maps["risk"].shape == (2, 8, 8)


def test_threshold_risk_map():
    risk = torch.tensor([[0.1, 0.5, 0.9]])
    mask = threshold_risk_map(risk, threshold=0.5)

    assert mask.tolist() == [[False, True, True]]
