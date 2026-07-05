"""Generate uncertainty tensors from saved segmentation logits or probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.research.uncertainty import logits_to_uncertainty, probabilities_to_uncertainty, threshold_risk_map


def ensure_batched(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected [C,H,W] or [B,C,H,W], received {tuple(tensor.shape)}")


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def generate(input_path: Path, output_dir: Path, input_type: str, risk_threshold: float) -> None:
    tensor = ensure_batched(torch.load(input_path, map_location="cpu"))

    if input_type == "logits":
        maps = logits_to_uncertainty(tensor)
    elif input_type == "probabilities":
        maps = probabilities_to_uncertainty(tensor)
    else:
        raise ValueError("input_type must be 'logits' or 'probabilities'")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_path": str(input_path),
        "input_type": input_type,
        "input_shape": list(tensor.shape),
        "risk_threshold": risk_threshold,
        "samples": [],
    }

    for index in range(tensor.shape[0]):
        sample_dir = output_dir / f"sample_{index:04d}"
        high_risk = threshold_risk_map(maps.risk[index], threshold=risk_threshold)

        save_tensor(sample_dir / "prediction.pt", maps.prediction[index])
        save_tensor(sample_dir / "confidence.pt", maps.confidence[index])
        save_tensor(sample_dir / "entropy.pt", maps.entropy[index])
        save_tensor(sample_dir / "normalized_entropy.pt", maps.normalized_entropy[index])
        save_tensor(sample_dir / "risk.pt", maps.risk[index])
        save_tensor(sample_dir / "high_risk_mask.pt", high_risk)

        metadata["samples"].append(
            {
                "index": index,
                "directory": str(sample_dir),
                "mean_confidence": float(maps.confidence[index].mean()),
                "mean_entropy": float(maps.entropy[index].mean()),
                "mean_risk": float(maps.risk[index].mean()),
                "high_risk_fraction": float(high_risk.float().mean()),
            }
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-type", choices=["logits", "probabilities"], default="logits")
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.input, args.output_dir, args.input_type, args.risk_threshold)
