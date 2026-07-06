# Continual Learning Metrics

This document introduces metrics for continual semantic segmentation experiments.

## Motivation

A model used in autonomous driving should adapt to new cities, weather, lighting, and datasets without losing performance on previous domains. Continual-learning evaluation measures this stability-plasticity trade-off.

## Performance matrix

The utilities expect a matrix with shape:

```text
[training_step, evaluation_domain]
```

Rows represent sequential training stages. Columns represent evaluation domains.

## Metrics

`src/research/continual_metrics.py` provides:

- `average_accuracy`: final average performance across domains,
- `forgetting_score`: how much performance dropped from previous best scores,
- `retained_performance`: final score relative to best historical score,
- `forward_transfer`: performance on future domains before direct training,
- `continual_learning_report`: compact summary dictionary.

## Research use

Use these metrics for protocols such as:

- CamVid to Cityscapes to BDD100K,
- clear weather to rain to fog to night,
- city-by-city adaptation,
- dataset-by-dataset adaptation.

## Important interpretation

High final accuracy is not sufficient. A good continual model should combine high average accuracy with low forgetting and strong retained performance.
