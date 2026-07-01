# Experiment Protocol

This document defines a reproducible experimental protocol for **Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving**.

The project evaluates semantic segmentation models for urban driving scenes using the CamVid dataset, with emphasis on safety-critical categories such as road, vehicles, sidewalks and pedestrians.

## 1. Research questions

The experimental design is guided by four core questions:

1. **Architecture effect:** how do CNN-based models compare with transformer-based segmentation models?
2. **Label-space effect:** does reducing CamVid from 12 classes to 5 safety-oriented classes improve performance and interpretability?
3. **Loss-function effect:** does weighted cross-entropy improve rare-class segmentation?
4. **Safety-critical performance:** how reliable are the models on dynamic objects, especially vehicles and pedestrians?

## 2. Dataset

The experiments use the **CamVid road-scene semantic segmentation dataset**.

### Reduced 5-class label space

| Reduced class | Autonomous-driving interpretation |
|---|---|
| Background | Non-critical or merged scene elements |
| Road | Drivable road surface |
| Sidewalk | Pedestrian-side navigable area |
| Vehicle | Cars and other road vehicles |
| Pedestrian | Human road users |

The reduced label space is motivated by autonomous-driving safety: the aim is not exhaustive scene parsing, but robust understanding of road layout and dynamic object classes.

## 3. Model configurations

| Model | Family | Main purpose |
|---|---|---|
| U-Net + ResNet34 encoder | CNN encoder-decoder | Strong baseline for semantic segmentation |
| DeepLabV3+ | CNN with atrous spatial pyramid pooling | Multi-scale context modeling |
| SegFormer MiT-B0 | Transformer-based | Lightweight transformer comparison |

## 4. Experimental setup

| Component | Configuration |
|---|---|
| Input resolution | 256 × 256 |
| Batch size | 4 |
| CNN optimizer | Adam |
| Transformer optimizer | AdamW |
| CNN learning rate | 1e-3 |
| Transformer learning rate | 5e-5 |
| Loss functions | Cross-Entropy, Weighted Cross-Entropy |
| Primary metric | Mean Intersection over Union |
| Secondary metric | Class-wise IoU |

## 5. Evaluation metrics

### Mean Intersection over Union

Mean Intersection over Union, or mIoU, measures the average overlap between predicted and ground-truth semantic regions across all classes.

### Class-wise IoU

Class-wise IoU is essential for autonomous-driving analysis because high global mIoU can hide poor performance on small but safety-critical classes, such as pedestrians.

### Qualitative overlays

Prediction masks and image overlays should be used to inspect:

- road-boundary quality,
- sidewalk separation,
- vehicle segmentation,
- pedestrian misses,
- class confusion between background and small objects.

## 6. Reported results

### Overall model comparison

| Model | Label setup | Loss | mIoU |
|---|---|---|---:|
| U-Net | 12 classes | Cross-Entropy | 0.4797 |
| U-Net | 5 classes | Cross-Entropy | **0.7240** |
| DeepLabV3+ | 5 classes | Cross-Entropy | 0.7132 |
| U-Net | 5 classes | Weighted Cross-Entropy | 0.6823 |
| SegFormer | 5 classes | Cross-Entropy | 0.6135 |

### Best model class-wise IoU

| Class | IoU |
|---|---:|
| Background | 0.9646 |
| Road | 0.9591 |
| Sidewalk | 0.8424 |
| Vehicle | 0.5944 |
| Pedestrian | 0.2596 |

## 7. Ablation studies

### 7.1 Label-space ablation

Compare U-Net trained on:

- original 12-class CamVid formulation,
- reduced 5-class safety-oriented formulation.

The current results show that reducing the label space improves mIoU from **0.4797** to **0.7240**. This suggests that task formulation is a major factor in segmentation performance.

### 7.2 Loss-function ablation

Compare U-Net with:

- standard cross-entropy,
- weighted cross-entropy.

The weighted setup did not outperform standard cross-entropy in the reported experiment. This should be discussed as a trade-off: class reweighting may increase rare-class attention but can reduce stability or dominant-class performance.

### 7.3 Architecture ablation

Compare:

- U-Net,
- DeepLabV3+,
- SegFormer.

The current ranking suggests that CNN-based models are stronger in this experimental setup. This does not rule out transformer models; it indicates that SegFormer may require more tuning, higher resolution, longer training or stronger augmentation.

## 8. Presentation structure

For a clear academic presentation, use this order:

1. **Motivation:** autonomous vehicles need reliable road and dynamic-object understanding.
2. **Dataset and labels:** explain CamVid and the 5-class safety-oriented reduction.
3. **Models:** introduce U-Net, DeepLabV3+ and SegFormer.
4. **Metrics:** explain mIoU and class-wise IoU.
5. **Results table:** show overall model comparison.
6. **Class-wise analysis:** emphasize strong road/sidewalk IoU and weak pedestrian IoU.
7. **Qualitative examples:** show masks and overlays.
8. **Limitations and future work:** higher resolution, temporal consistency and improved rare-class losses.

## 9. Recommended next experiments

- Train at higher resolution to improve pedestrian and small-object segmentation.
- Evaluate Focal Loss, Dice Loss and hybrid Dice + Cross-Entropy.
- Add precision/recall/F1 for vehicle and pedestrian classes.
- Add inference-time benchmarking in FPS and milliseconds per frame.
- Add temporal consistency evaluation for video sequences.
- Tune SegFormer learning rate, scheduler, augmentation and training length.
- Compare results on another driving dataset such as Cityscapes, if available.

## 10. Reproducibility checklist

Future experiment logs should record:

- dataset split,
- random seed,
- number of epochs,
- optimizer and scheduler,
- augmentation pipeline,
- input resolution,
- hardware/GPU,
- checkpoint name,
- final mIoU,
- per-class IoU,
- qualitative output directory.
