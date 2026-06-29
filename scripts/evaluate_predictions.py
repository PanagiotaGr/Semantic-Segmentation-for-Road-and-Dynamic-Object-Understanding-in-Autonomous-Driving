"""Evaluate saved segmentation masks with IoU, precision, recall, F1, and confusion matrix.

Expected input format:
- ground-truth masks: integer class-index PNG files
- prediction masks: integer class-index PNG files with matching filenames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.advanced_metrics import (
    confusion_matrix_from_masks,
    metrics_from_confusion_matrix,
    normalized_confusion_matrix,
)


CLASS_NAMES = ["Background", "Road", "Sidewalk", "Vehicle", "Pedestrian"]


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.int64)


def save_confusion_heatmap(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest")
    axis.set_title("Normalized confusion matrix")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results/evaluation"), type=Path)
    parser.add_argument("--num-classes", default=5, type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

    gt_files = sorted(args.gt_dir.glob("*.png"))
    if not gt_files:
        raise FileNotFoundError(f"No PNG masks found in {args.gt_dir}")

    for gt_path in gt_files:
        pred_path = args.pred_dir / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction for {gt_path.name}")
        matrix += confusion_matrix_from_masks(
            load_mask(gt_path),
            load_mask(pred_path),
            num_classes=args.num_classes,
        )

    class_names = CLASS_NAMES[: args.num_classes]
    report = metrics_from_confusion_matrix(matrix, class_names)
    norm_matrix = normalized_confusion_matrix(matrix)

    report.table.to_csv(args.output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(args.output_dir / "confusion_matrix.csv")
    pd.DataFrame(norm_matrix, index=class_names, columns=class_names).to_csv(args.output_dir / "confusion_matrix_normalized.csv")
    save_confusion_heatmap(norm_matrix, class_names, args.output_dir / "confusion_matrix.png")

    print(report.table.to_string(index=False))
    print(f"Mean IoU: {report.mean_iou:.4f}")
    print(f"Macro F1: {report.macro_f1:.4f}")


if __name__ == "__main__":
    main()
