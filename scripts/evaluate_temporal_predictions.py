"""Evaluate temporal stability from saved predicted masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.research.temporal_metrics import temporal_stability_score


def evaluate(input_path: Path, output_path: Path, num_classes: int) -> None:
    predictions = torch.load(input_path, map_location="cpu")
    if predictions.ndim != 3:
        raise ValueError("Expected predictions with shape [frames, height, width]")

    scores = temporal_stability_score(predictions.long(), num_classes=num_classes)
    report = {name: float(value.detach().cpu()) for name, value in scores.items()}
    report["input_path"] = str(input_path)
    report["input_shape"] = list(predictions.shape)
    report["num_classes"] = num_classes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-classes", required=True, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.input, args.output, args.num_classes)
