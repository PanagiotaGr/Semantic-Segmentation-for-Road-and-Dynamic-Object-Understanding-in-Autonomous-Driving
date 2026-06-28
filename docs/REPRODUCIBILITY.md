# Reproducibility Guide

This document summarizes the information needed to reproduce and extend the semantic segmentation experiments.

## Environment

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

The project uses the CamVid dataset. A typical local directory layout is:

```text
data/
└── CamVid/
    ├── train/
    ├── train_labels/
    ├── val/
    ├── val_labels/
    ├── test/
    └── test_labels/
```

The exact folder names may be adapted to match the training scripts.

## Label Mapping

The reduced 5-class label space is:

| Class index | Class name |
| ---: | --- |
| 0 | Background |
| 1 | Road |
| 2 | Sidewalk |
| 3 | Vehicle |
| 4 | Pedestrian |

The mapping from the original CamVid labels should be kept fixed across all experiments so that model comparisons remain fair.

## Recommended Experiment Log

For each experiment, record:

| Field | Example |
| --- | --- |
| Model | U-Net / DeepLabV3+ / SegFormer |
| Backbone | ResNet34 / MiT-B0 |
| Label setup | 12-class / 5-class |
| Loss | CE / Weighted CE |
| Optimizer | Adam / AdamW |
| Learning rate | 1e-3 / 5e-5 |
| Epochs | e.g., 25 |
| Batch size | 4 |
| Image size | 256 × 256 |
| Random seed | e.g., 42 |
| mIoU | final score |
| Notes | qualitative observations |

## Evaluation

Report both global and class-level performance:

* Mean Intersection over Union (mIoU)
* Per-class IoU
* Qualitative prediction masks
* Overlay visualizations

## Notes for Future Extensions

To improve the scientific strength of the project, future experiments should include:

* multiple random seeds,
* validation curves,
* confusion matrix,
* precision and recall per class,
* inference-time benchmark,
* higher-resolution input experiments,
* stronger loss functions for rare classes.
