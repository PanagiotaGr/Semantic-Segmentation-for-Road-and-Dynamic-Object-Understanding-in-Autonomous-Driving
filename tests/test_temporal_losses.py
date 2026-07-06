import torch

from src.research.temporal_losses import (
    combined_segmentation_temporal_loss,
    temporal_logit_consistency_loss,
    temporal_probability_consistency_loss,
)


def test_temporal_probability_consistency_loss_zero_for_identical_frames():
    frame = torch.randn(2, 1, 5, 8, 8)
    logits = frame.repeat(1, 3, 1, 1, 1)
    loss = temporal_probability_consistency_loss(logits)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_temporal_logit_consistency_loss_positive_for_different_frames():
    logits = torch.randn(2, 3, 5, 8, 8)
    loss = temporal_logit_consistency_loss(logits)
    assert loss.item() >= 0.0


def test_temporal_probability_consistency_loss_none_reduction_shape():
    logits = torch.randn(2, 4, 5, 8, 8)
    loss = temporal_probability_consistency_loss(logits, reduction="none")
    assert loss.shape == (2, 3, 8, 8)


def test_combined_segmentation_temporal_loss_adds_penalty():
    segmentation_loss = torch.tensor(1.0)
    logits = torch.randn(2, 3, 5, 8, 8)
    combined = combined_segmentation_temporal_loss(segmentation_loss, logits, temporal_weight=0.5)
    assert combined.item() >= segmentation_loss.item()
