# Model Ablation Study

This document describes the expanded model-comparison plan for the project.

## Goal

The goal is to compare several semantic segmentation architectures under the same dataset, label space, preprocessing, loss function, and evaluation protocol.

## Models

| Model | Family | Reason for inclusion |
| --- | --- | --- |
| U-Net | Encoder-decoder CNN | Strong baseline and best current model |
| DeepLabV3+ | Atrous CNN | Strong context-aware segmentation model |
| DeepLabV3 | Atrous CNN | Simpler DeepLab comparison |
| PSPNet | Pyramid pooling CNN | Multi-scale scene context |
| FPN | Feature pyramid CNN | Multi-scale feature fusion |
| PAN | Attention-based decoder | Efficient segmentation comparison |
| LinkNet | Encoder-decoder CNN | Lightweight decoder baseline |
| MAnet | Attention segmentation CNN | Attention-based segmentation comparison |
| SegFormer | Transformer | Transformer-based comparison already included |

## Controlled Variables

To make the comparison fair, keep these fixed:

* dataset split,
* reduced 5-class label space,
* image resolution,
* batch size,
* optimizer,
* learning rate policy,
* number of epochs or early stopping rule,
* random seed,
* evaluation metrics.

## Metrics to Report

Each model should report:

* mIoU,
* class-wise IoU,
* precision per class,
* recall per class,
* F1-score per class,
* normalized confusion matrix,
* inference FPS.

## Recommended Table

| Model | Encoder | mIoU | Vehicle IoU | Pedestrian IoU | Macro F1 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | ResNet34 | TBD | TBD | TBD | TBD | TBD |
| DeepLabV3+ | ResNet34 | TBD | TBD | TBD | TBD | TBD |
| PSPNet | ResNet34 | TBD | TBD | TBD | TBD | TBD |
| FPN | ResNet34 | TBD | TBD | TBD | TBD | TBD |
| PAN | ResNet34 | TBD | TBD | TBD | TBD | TBD |
| LinkNet | ResNet34 | TBD | TBD | TBD | TBD | TBD |
| MAnet | ResNet34 | TBD | TBD | TBD | TBD | TBD |
| SegFormer | MiT-B0 | TBD | TBD | TBD | TBD | TBD |

## Interpretation Strategy

The best model should not be selected only by global mIoU. For autonomous-driving perception, pedestrian and vehicle performance must be discussed separately because they are safety-critical dynamic classes.
