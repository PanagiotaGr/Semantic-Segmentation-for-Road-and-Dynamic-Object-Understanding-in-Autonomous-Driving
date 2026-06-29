"""Utility script for measuring segmentation inference speed.

This script is intentionally model-agnostic. Import or create your trained model
inside the build_model function, then run the benchmark on CPU or GPU.
"""

from __future__ import annotations

import argparse
import time

import torch


def build_model() -> torch.nn.Module:
    """Replace this placeholder with the trained segmentation model."""
    raise NotImplementedError("Load your trained model inside build_model().")


def benchmark(model: torch.nn.Module, image_size: int, device: str, warmup: int, steps: int) -> dict[str, float]:
    model = model.to(device)
    model.eval()

    sample = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(steps):
            _ = model(sample)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    seconds_per_image = elapsed / steps
    fps = 1.0 / seconds_per_image
    return {"seconds_per_image": seconds_per_image, "fps": fps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark segmentation model inference speed.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    model = build_model()
    result = benchmark(model, args.image_size, args.device, args.warmup, args.steps)

    print(f"Device: {args.device}")
    print(f"Image size: {args.image_size} x {args.image_size}")
    print(f"Seconds per image: {result['seconds_per_image']:.6f}")
    print(f"FPS: {result['fps']:.2f}")


if __name__ == "__main__":
    main()
