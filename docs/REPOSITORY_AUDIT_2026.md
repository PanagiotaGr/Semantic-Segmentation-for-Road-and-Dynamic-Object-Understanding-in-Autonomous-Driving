# Repository Research Audit — 2026

## Purpose

This audit records the current state of the project and separates completed evidence from implemented utilities and proposed research extensions.

## Current strengths

- Reproducible baseline direction on CamVid.
- Comparison of U-Net, DeepLabV3+, and SegFormer.
- Safety-oriented five-class label mapping.
- Class-wise analysis rather than relying only on global mIoU.
- Implemented uncertainty, temporal, and continual-learning utilities.
- Honest documentation of limitations.

## Main research risk

The repository currently contains several model-specific scripts and advanced research utilities, but it does not yet expose one unified end-to-end experiment interface. This makes it difficult to guarantee that all models are trained, evaluated, logged, and compared under identical conditions.

## Priority gaps

### 1. Unified experiment pipeline

Required:

- one configuration-driven training entry point;
- one evaluation entry point;
- shared dataset and label-mapping code;
- common checkpoint format;
- automatic experiment metadata;
- deterministic seed handling.

### 2. Small-object and pedestrian performance

The main safety weakness is pedestrian segmentation. Future experiments should prioritize pedestrian recall, pedestrian IoU, boundary quality, and failure analysis rather than optimizing only global mIoU.

### 3. Reliability evaluation

Uncertainty utilities must be connected to real model errors through calibration metrics, error-detection metrics, and risk-coverage analysis.

### 4. Temporal validation

Temporal utilities should be evaluated in a complete sequence-based experiment. Implemented components must not be described as experimentally validated until a configuration, checkpoint, and result report exist.

### 5. Robustness and deployment

The repository still requires controlled corruption tests, cross-domain evaluation, latency measurement, memory measurement, and export validation.

## Truthfulness states

Every feature and experiment should use exactly one of these states:

- **Implemented:** code exists and has automated tests or a verified execution path.
- **Experimentally validated:** a recorded configuration, checkpoint, environment, metrics, and report exist.
- **Proposed:** planned work with no completed evidence yet.

## Recommended implementation order

1. Standardize experiment configuration and metadata.
2. Reproduce the strongest five-class U-Net baseline.
3. Add class-imbalance and small-object ablations.
4. Add calibration and uncertainty evaluation.
5. Add temporal baselines.
6. Add robustness benchmarks.
7. Add ONNX/runtime benchmarking.
8. Connect verified results to thesis chapters.

## Definition of research-ready

An experiment is research-ready only when it includes:

- source commit;
- dataset split and mapping;
- full configuration;
- random seed;
- environment information;
- checkpoint identifier;
- global and class-wise metrics;
- qualitative failure examples;
- limitations;
- machine-readable output.
