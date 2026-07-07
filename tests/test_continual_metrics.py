import torch

from src.research.continual_metrics import average_accuracy, continual_learning_report


def test_average_accuracy():
    scores = torch.tensor([[0.8, 0.0], [0.6, 0.7]])
    value = average_accuracy(scores)
    assert torch.isclose(value, torch.tensor(0.65))


def test_report_keys():
    scores = torch.tensor([[0.8, 0.0], [0.6, 0.7]])
    report = continual_learning_report(scores)
    assert "average_accuracy" in report
    assert "forgetting_score" in report
    assert "retained_performance" in report
    assert "forward_transfer" in report
