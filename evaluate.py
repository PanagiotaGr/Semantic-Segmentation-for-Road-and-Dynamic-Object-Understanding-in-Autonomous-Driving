"""Evaluate a trained segmentation checkpoint on a configured split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.framework.config import load_experiment_config
from src.framework.data import SegmentationFolderDataset
from src.framework.losses import build_loss
from src.framework.models import build_model
from src.framework.training import run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="evaluation")
    args = parser.parse_args()

    config = load_experiment_config(args.config).raw
    data_cfg = config["data"]
    split = str(data_cfg.get(f"{args.split}_split", args.split))
    dataset = SegmentationFolderDataset(
        Path(data_cfg["root"]), split, tuple(int(v) for v in data_cfg["image_size"])
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    criterion = build_loss(config).to(device)
    metrics = run_epoch(model, loader, criterion, device, int(data_cfg["num_classes"]))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    names = data_cfg.get("class_names", [str(i) for i in range(len(metrics["class_iou"]))])
    with (output / "class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "iou"])
        writer.writerows(zip(names, metrics["class_iou"]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
