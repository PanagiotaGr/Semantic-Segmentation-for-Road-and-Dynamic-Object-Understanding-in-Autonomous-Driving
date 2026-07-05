# Baseline Module

This module will contain reproducible baseline experiments for semantic segmentation in autonomous-driving scenes.

## Purpose

The baseline module should answer a controlled scientific question:

> How do architecture, input resolution, loss function, augmentation, and label-space formulation affect safety-critical road-scene segmentation?

## Initial Models

- U-Net with ResNet34 encoder
- DeepLabV3+
- SegFormer MiT-B0

## Planned Extensions

- higher-resolution training,
- stronger augmentation,
- class-balanced crop sampling,
- focal/dice/tversky/lovasz losses,
- inference speed benchmarking,
- standardized qualitative failure analysis.

## Required Outputs

Each run should produce:

- `metrics.json`
- `class_iou.csv`
- `config.yaml`
- `training_curve.png`
- `qualitative_examples/`
- `failure_cases/`

## Minimum Reporting Fields

- model name
- dataset
- split
- input resolution
- optimizer
- learning rate
- batch size
- number of epochs
- random seed
- hardware
- training time
- inference FPS
- mIoU
- class-wise IoU
- pedestrian IoU
- vehicle IoU
