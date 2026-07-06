import torch

from src.research.temporal_metrics import (
    consecutive_frame_iou,
    frame_change_rate,
    temporal_stability_score,
)


def test_frame_change_rate_zero_for_identical_frames():
    predictions = torch.zeros(3, 4, 4, dtype=torch.long)
    assert frame_change_rate(predictions).item() == 0.0


def test_frame_change_rate_detects_changes():
    predictions = torch.zeros(2, 2, 2, dtype=torch.long)
    predictions[1, 0, 0] = 1
    assert frame_change_rate(predictions).item() == 0.25


def test_consecutive_frame_iou_identical_frames():
    predictions = torch.zeros(3, 4, 4, dtype=torch.long)
    assert consecutive_frame_iou(predictions, num_classes=2).item() == 1.0


def test_temporal_stability_score_keys():
    predictions = torch.zeros(3, 4, 4, dtype=torch.long)
    score = temporal_stability_score(predictions, num_classes=2)
    assert "frame_change_rate" in score
    assert "consecutive_frame_iou" in score
    assert "stability" in score
