# Experimental Protocol

This document describes the experimental design used for semantic segmentation of road-scene images.

## Objective

The objective is to evaluate how model architecture, label-space design, and loss-function selection affect semantic segmentation performance in autonomous-driving scene understanding.

## Research Questions

1. Does reducing the original CamVid label space improve segmentation performance?
2. Do CNN-based architectures outperform transformer-based models in this dataset and resolution setting?
3. Does weighted cross-entropy improve rare-class segmentation?
4. Which classes remain most difficult and why?

## Compared Models

| Model | Family | Main role in the study |
| --- | --- | --- |
| U-Net with ResNet34 encoder | CNN encoder-decoder | Strong baseline and best overall model |
| DeepLabV3+ | CNN with atrous spatial context | Competitive model for structured road scenes |
| SegFormer MiT-B0 | Transformer-based segmentation | Lightweight transformer comparison |

## Label-Space Comparison

Two label formulations are compared:

| Setup | Description |
| --- | --- |
| 12-class setup | Original CamVid-style semantic classes |
| 5-class setup | Reduced safety-critical formulation |

The 5-class setup focuses on Background, Road, Sidewalk, Vehicle, and Pedestrian.

## Loss Functions

| Loss | Motivation |
| --- | --- |
| Cross-Entropy | Standard multi-class segmentation objective |
| Weighted Cross-Entropy | Attempts to compensate for class imbalance |

## Metrics

The main metric is mean Intersection over Union. Class-wise IoU is also reported to identify failure modes in rare and safety-critical classes.

## Reported Results

| Model | Setup | mIoU |
| --- | --- | ---: |
| U-Net | 12 classes | 0.4797 |
| U-Net | 5 classes | 0.7240 |
| DeepLabV3+ | 5 classes | 0.7132 |
| U-Net with weighted loss | 5 classes | 0.6823 |
| SegFormer | 5 classes | 0.6135 |

## Interpretation

The best result is achieved by U-Net under the reduced 5-class setup. This indicates that a carefully designed label space can be more important than simply increasing architectural complexity.

DeepLabV3+ remains competitive and is useful for comparison because its atrous spatial pyramid design can capture broader context.

SegFormer does not outperform the CNN models in this configuration. This may be related to limited dataset size, reduced image resolution, and the need for more extensive transformer tuning.

## Main Failure Case

Pedestrian segmentation remains the most difficult class. This is expected because pedestrians are usually small, visually variable, and under-represented compared with road and background pixels.

## Recommended Additional Experiments

* Train with higher image resolution.
* Add Focal Loss and Dice Loss.
* Evaluate multiple random seeds.
* Add precision, recall, and confusion matrix per class.
* Measure inference speed in frames per second.
* Compare with Cityscapes or another larger urban-scene dataset.
