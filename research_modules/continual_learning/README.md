# Continual Learning Module

This module studies how semantic-segmentation models can adapt to new cities, weather conditions, and datasets without forgetting previous domains.

## Research Question

Can an autonomous-driving segmentation model learn continuously from new environments while preserving performance on previously learned environments?

## Motivation

Autonomous vehicles encounter changing cities, roads, lighting, weather, sensors, and driving cultures. A static model trained once may not remain reliable. However, naive fine-tuning often causes catastrophic forgetting.

## Continual-Learning Protocols

### Dataset Sequence

Example:

```text
CamVid -> Cityscapes -> BDD100K -> ACDC -> Mapillary
```

### Weather Sequence

Example:

```text
clear -> rain -> fog -> snow -> night
```

### City Sequence

Example:

```text
city_1 -> city_2 -> city_3 -> city_4
```

## Candidate Methods

1. **Naive fine-tuning**
   - baseline for catastrophic forgetting.

2. **Replay memory**
   - store a small set of previous examples.

3. **Knowledge distillation**
   - preserve predictions from an older teacher model.

4. **Regularization-based learning**
   - penalize destructive parameter changes.

5. **Adapters**
   - keep domain-specific lightweight modules.

6. **LoRA**
   - parameter-efficient domain adaptation.

7. **Pseudo-labeling**
   - adapt with unlabeled data from new domains.

## Metrics

- average mIoU across domains
- current-domain mIoU
- previous-domain retained mIoU
- forgetting score
- backward transfer
- forward transfer
- parameter growth
- memory size

## Expected Outputs

- continual-learning benchmark table
- forgetting curves
- domain-transfer matrix
- retained-performance plots
- qualitative comparison across domains

## Expected Contribution

The module should establish a rigorous protocol for lifelong autonomous-driving segmentation and evaluate whether parameter-efficient adaptation can reduce forgetting while keeping the model deployable.
