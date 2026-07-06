# Temporal Fusion Experiment Recipe

This document describes the first complete temporal baseline recipe.

## Goal

Compare frame-independent semantic segmentation against a simple temporal feature-fusion baseline.

## Components

The recipe connects previously added utilities:

- temporal sequence dataset,
- temporal feature fusion,
- temporal consistency loss,
- temporal stability metrics,
- temporal difference maps.

## Configuration

See:

```text
configs/temporal_fusion_baseline.yaml
```

## Baseline design

The model should encode each frame independently using the existing single-frame encoder, fuse frame-level features, and predict a segmentation mask for the center or last frame.

## Recommended ablations

- sequence length: 2, 3, 5,
- fusion mode: mean, sum, concat,
- temporal loss weight: 0.0, 0.05, 0.1, 0.2,
- target frame: center versus last.

## Expected reporting

Report both segmentation quality and temporal stability:

- mIoU,
- class-wise IoU,
- pedestrian IoU,
- vehicle IoU,
- frame change rate,
- consecutive-frame IoU,
- temporal stability,
- temporal change frequency.
