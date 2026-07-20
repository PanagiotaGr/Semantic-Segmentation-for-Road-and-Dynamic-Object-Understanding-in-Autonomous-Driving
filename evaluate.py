"""Evaluate a trained segmentation checkpoint on a configured split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.framework.config import load_experiment_config
from src.framework.data import build_dataset
from src.framework.losses import build_loss
from src.framework.models import build_model
from src.framework.reporting import concatenate_predictions, save_confusion_matrix_plot
from src.framework.research_metrics import boundary_iou, class_binary_metrics
from src.framework.training import run_epoch
from src.framework.visualization import save_prediction_overlay


def export_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    limit: int,
) -> None:
    """Write qualitative prediction panels for the first evaluation samples."""
    model.eval()
    written = 0
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            for image, prediction in zip(images, predictions):
                save_prediction_overlay(
                    image,
                    prediction,
                    output_dir / f"prediction_{written:04d}.png",
                )
                written += 1
                if written >= limit:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--prediction-limit", type=int, default=16)
    parser.add_argument("--boundary-dilation", type=int, default=1)
    args = parser.parse_args()

    config = load_experiment_config(args.config).raw
    data_cfg = config["data"]
    split = str(data_cfg.get(f"{args.split}_split", args.split))
    dataset = build_dataset(config, split, training=False)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    criterion = build_loss(config).to(device)
    metrics = run_epoch(model, loader, criterion, device, int(data_cfg["num_classes"]))

    predictions, targets = concatenate_predictions(model, loader, device)
    ignore_index = int(data_cfg.get("ignore_index", 255))
    pedestrian_class_id = int(data_cfg.get("pedestrian_class_id", 4))
    metrics["boundary_iou"] = boundary_iou(
        predictions,
        targets,
        ignore_index=ignore_index,
        dilation=args.boundary_dilation,
    )
    metrics["pedestrian"] = class_binary_metrics(
        predictions,
        targets,
        class_id=pedestrian_class_id,
        ignore_index=ignore_index,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    names = data_cfg.get("class_names", [str(i) for i in range(len(metrics["class_iou"]))])
    with (output / "class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "iou", "precision", "recall"])
        writer.writerows(
            zip(
                names,
                metrics["class_iou"],
                metrics.get("class_precision", []),
                metrics.get("class_recall", []),
            )
        )
    with (output / "pedestrian_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics["pedestrian"].items())

    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        names,
        output / "confusion_matrix.png",
        normalize=True,
    )
    if args.prediction_limit > 0:
        export_predictions(model, loader, device, output / "predictions", args.prediction_limit)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
