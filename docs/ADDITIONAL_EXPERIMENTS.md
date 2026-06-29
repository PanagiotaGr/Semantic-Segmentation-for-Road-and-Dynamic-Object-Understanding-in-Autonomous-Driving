# Additional Experiments

This document proposes extra experiments that strengthen the scientific evaluation of the project.

## 1. Loss Function Ablation

Compare different loss functions under the same model and label setup.

| Experiment | Model | Loss | Goal |
| --- | --- | --- | --- |
| CE baseline | U-Net | Cross-Entropy | Baseline comparison |
| Weighted CE | U-Net | Weighted Cross-Entropy | Class imbalance handling |
| Focal Loss | U-Net | Focal Loss | Focus on hard examples |
| Dice Loss | U-Net | Dice Loss | Improve overlap quality |
| CE + Dice | U-Net | Combined Loss | Balance pixel accuracy and region overlap |

Expected value: This experiment directly tests whether rare-class segmentation, especially pedestrian IoU, can be improved.

## 2. Resolution Ablation

Train the same model at different input resolutions.

| Experiment | Image size | Motivation |
| --- | ---: | --- |
| Low resolution | 128 x 128 | Fast baseline |
| Current setup | 256 x 256 | Main reported setting |
| Higher resolution | 384 x 384 | Better small-object detail |
| High resolution | 512 x 512 | Stronger pedestrian and boundary segmentation |

Expected value: Higher resolution should help small objects, but may increase memory usage and training time.

## 3. Random Seed Stability

Repeat the best model with multiple random seeds.

| Run | Seed |
| --- | ---: |
| Run 1 | 0 |
| Run 2 | 1 |
| Run 3 | 2 |
| Run 4 | 42 |
| Run 5 | 123 |

Report mean and standard deviation of mIoU. This makes the results more statistically reliable.

## 4. Class-Specific Error Analysis

For each class, report:

* IoU,
* precision,
* recall,
* false positives,
* false negatives,
* most common confusion pairs.

This is especially useful for the pedestrian and vehicle classes.

## 5. Inference Speed Benchmark

Measure average inference time and frames per second.

| Model | Device | Image size | FPS | Notes |
| --- | --- | ---: | ---: | --- |
| U-Net | GPU or CPU | 256 x 256 | TBD | Best mIoU model |
| DeepLabV3+ | GPU or CPU | 256 x 256 | TBD | Competitive CNN model |
| SegFormer | GPU or CPU | 256 x 256 | TBD | Transformer comparison |

Expected value: This connects segmentation quality with real-time autonomous-driving constraints.

## 6. Qualitative Failure Cases

Create a folder with examples where the model fails. Useful categories:

* missed pedestrians,
* vehicle-boundary errors,
* sidewalk-road confusion,
* shadows or lighting errors,
* occlusions.

This improves the discussion section because it connects metrics with visual evidence.

## Priority Recommendation

The strongest next experiments are:

1. Focal Loss and Dice Loss for pedestrian improvement.
2. Resolution ablation for small-object segmentation.
3. Multi-seed evaluation for statistical reliability.
4. Inference speed benchmark for real-time relevance.
