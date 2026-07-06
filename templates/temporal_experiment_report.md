# Temporal Experiment Report Template

Use this template for sequence-based semantic segmentation experiments.

## Experiment ID

`TEMP-YYYY-MM-DD-001`

## Research Question

Does temporal context improve segmentation quality and stability compared with frame-independent prediction?

## Hypothesis

State whether the temporal method is expected to improve mIoU, reduce flicker, improve dynamic-object stability, or trade accuracy for smoother predictions.

## Dataset and Sequence Setup

- dataset:
- split:
- sequence length:
- frame stride:
- target frame: center / last
- image resolution:
- label mapping:

## Model

- frame encoder:
- temporal fusion method:
- segmentation decoder/head:
- number of parameters:
- pretrained weights:

## Training Configuration

- batch size:
- epochs:
- optimizer:
- learning rate:
- scheduler:
- segmentation loss:
- temporal consistency loss:
- temporal loss weight:
- augmentations:
- random seed:
- hardware:

## Segmentation Metrics

| Metric | Value |
| --- | ---: |
| mIoU | |
| Background IoU | |
| Road IoU | |
| Sidewalk IoU | |
| Vehicle IoU | |
| Pedestrian IoU | |
| Pixel Accuracy | |

## Temporal Metrics

| Metric | Value |
| --- | ---: |
| Frame Change Rate | |
| Consecutive-Frame IoU | |
| Temporal Stability | |
| Mean Change Frequency | |
| Dynamic-Object Stability | |

## Comparison Against Frame Baseline

| Model | mIoU | Vehicle IoU | Pedestrian IoU | Frame Change Rate | Temporal Stability |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frame baseline | | | | | |
| Temporal model | | | | | |

## Qualitative Analysis

Include paths to:

- input frame sequence,
- predicted mask sequence,
- temporal difference maps,
- change-frequency map,
- failure cases,
- video overlay if available.

## Failure Analysis

Discuss:

- flickering boundaries,
- pedestrian instability,
- vehicle instability,
- false smoothing,
- temporal lag,
- cases where temporal fusion worsened predictions.

## Interpretation

Explain whether the method truly improves temporal perception or only smooths outputs.

## Limitations

List limitations in data, labels, temporal alignment, computational cost, or evaluation.

## Next Experiment

Recommended next ablation or model change.
