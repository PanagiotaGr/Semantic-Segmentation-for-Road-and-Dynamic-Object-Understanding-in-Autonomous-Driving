"""Generate temporal difference maps from saved predicted masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.research.temporal_diff import (
    temporal_change_count,
    temporal_change_frequency,
    temporal_change_maps,
)


def generate(input_path: Path, output_dir: Path) -> None:
    predictions = torch.load(input_path, map_location="cpu")
    if predictions.ndim != 3:
        raise ValueError("Expected predictions with shape [frames, height, width]")

    maps = temporal_change_maps(predictions.long())
    count = temporal_change_count(predictions.long())
    frequency = temporal_change_frequency(predictions.long())

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(maps.cpu(), output_dir / "change_maps.pt")
    torch.save(count.cpu(), output_dir / "change_count.pt")
    torch.save(frequency.cpu(), output_dir / "change_frequency.pt")

    report = {
        "input_path": str(input_path),
        "input_shape": list(predictions.shape),
        "change_maps_shape": list(maps.shape),
        "mean_change_frequency": float(frequency.mean()),
        "max_change_count": int(count.max()),
    }

    with (output_dir / "temporal_difference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.input, args.output_dir)
