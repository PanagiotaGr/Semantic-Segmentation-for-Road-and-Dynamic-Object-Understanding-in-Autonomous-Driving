"""Create a benchmark table for the reported segmentation experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS = [
    {"Model": "U-Net", "Backbone": "ResNet34", "Label setup": "12 classes", "Loss": "Cross-Entropy", "mIoU": 0.4797},
    {"Model": "U-Net", "Backbone": "ResNet34", "Label setup": "5 classes", "Loss": "Cross-Entropy", "mIoU": 0.7240},
    {"Model": "DeepLabV3+", "Backbone": "CNN encoder", "Label setup": "5 classes", "Loss": "Cross-Entropy", "mIoU": 0.7132},
    {"Model": "U-Net", "Backbone": "ResNet34", "Label setup": "5 classes", "Loss": "Weighted Cross-Entropy", "mIoU": 0.6823},
    {"Model": "SegFormer", "Backbone": "MiT-B0", "Label setup": "5 classes", "Loss": "Cross-Entropy", "mIoU": 0.6135},
]


def main() -> None:
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    frame = pd.DataFrame(RESULTS).sort_values("mIoU", ascending=False)
    markdown = "# Benchmark Summary\n\n"
    markdown += "This table summarizes the reported segmentation experiments.\n\n"
    markdown += frame.to_markdown(index=False)
    markdown += "\n"
    (output_dir / "BENCHMARKS.md").write_text(markdown, encoding="utf-8")
    frame.to_csv(output_dir / "benchmarks.csv", index=False)


main()
