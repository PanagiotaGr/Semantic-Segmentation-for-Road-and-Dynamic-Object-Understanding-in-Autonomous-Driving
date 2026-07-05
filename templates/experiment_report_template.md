# Experiment Report Template

Use this template for every experiment so results remain comparable and reproducible.

## Experiment ID

`EXP-YYYY-MM-DD-001`

## Research Question

What scientific question does this experiment answer?

## Hypothesis

What do we expect to happen, and why?

## Dataset

- dataset name:
- version:
- train split:
- validation split:
- test split:
- preprocessing:
- label mapping:

## Model

- architecture:
- encoder/backbone:
- decoder:
- pretrained weights:
- number of parameters:

## Training Configuration

- input resolution:
- batch size:
- epochs:
- optimizer:
- learning rate:
- scheduler:
- loss function:
- augmentations:
- random seed:
- hardware:

## Evaluation Metrics

| Metric | Value |
| --- | ---: |
| mIoU | |
| Background IoU | |
| Road IoU | |
| Sidewalk IoU | |
| Vehicle IoU | |
| Pedestrian IoU | |
| Pixel Accuracy | |
| Inference FPS | |

## Safety-Critical Analysis

Discuss:

- pedestrian false negatives,
- vehicle false negatives,
- road/sidewalk confusion,
- boundary errors,
- uncertainty or confidence failures if available.

## Qualitative Results

Add links or paths to:

- best predictions,
- worst predictions,
- failure cases,
- overlays,
- uncertainty maps if available.

## Comparison Against Baseline

| Model | mIoU | Vehicle IoU | Pedestrian IoU | Notes |
| --- | ---: | ---: | ---: | --- |
| Baseline | | | | |
| Current experiment | | | | |

## Interpretation

Explain what the result means scientifically. Avoid only reporting numbers.

## Limitations

List anything that may affect the validity of the result.

## Next Experiment

What should be tested next?
