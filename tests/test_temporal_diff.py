import torch

from src.research.temporal_diff import (
    temporal_change_count,
    temporal_change_frequency,
    temporal_change_maps,
)


def test_temporal_change_maps_shape():
    predictions = torch.zeros(3, 4, 4, dtype=torch.long)
    maps = temporal_change_maps(predictions)
    assert maps.shape == (2, 4, 4)


def test_temporal_change_count_detects_pixel_changes():
    predictions = torch.zeros(3, 2, 2, dtype=torch.long)
    predictions[1, 0, 0] = 1
    predictions[2, 0, 0] = 2
    count = temporal_change_count(predictions)
    assert count[0, 0].item() == 2
    assert count[1, 1].item() == 0


def test_temporal_change_frequency_range():
    predictions = torch.zeros(3, 2, 2, dtype=torch.long)
    predictions[1, 0, 0] = 1
    frequency = temporal_change_frequency(predictions)
    assert frequency.min().item() >= 0.0
    assert frequency.max().item() <= 1.0
