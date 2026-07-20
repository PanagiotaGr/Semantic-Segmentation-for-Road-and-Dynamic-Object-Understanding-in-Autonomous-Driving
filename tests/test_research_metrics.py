import torch

from src.framework.research_metrics import boundary_iou, class_binary_metrics


def test_boundary_iou_is_one_for_identical_masks():
    mask = torch.tensor([[[0, 0, 1], [0, 1, 1], [2, 2, 1]]])
    assert boundary_iou(mask, mask) == 1.0


def test_pedestrian_metrics_count_missed_pixels():
    target = torch.tensor([[[4, 4], [0, 0]]])
    prediction = torch.tensor([[[4, 0], [0, 0]]])
    metrics = class_binary_metrics(prediction, target, class_id=4)
    assert metrics["true_positive_pixels"] == 1
    assert metrics["false_negative_pixels"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["miss_rate"] == 0.5


def test_unknown_pixels_are_ignored():
    target = torch.tensor([[[255, 4]]])
    prediction = torch.tensor([[[4, 4]]])
    metrics = class_binary_metrics(prediction, target, class_id=4, ignore_index=255)
    assert metrics["true_positive_pixels"] == 1
    assert metrics["false_positive_pixels"] == 0
