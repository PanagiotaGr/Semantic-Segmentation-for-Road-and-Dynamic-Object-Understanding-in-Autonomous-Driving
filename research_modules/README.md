# Research Modules

This directory defines the future modular research architecture. The current repository can remain a semantic-segmentation baseline while new PhD-level capabilities are added incrementally.

## Module 1: Robust Baselines

Purpose:

- reproduce existing segmentation results,
- standardize experiment logging,
- compare CNN and transformer models,
- evaluate loss functions and image resolutions.

Expected files:

- `baselines/`
- `configs/baseline_*.yaml`
- `scripts/run_baseline_experiments.py`

## Module 2: Foundation Backbones

Purpose:

- integrate pretrained visual encoders,
- compare frozen, partially fine-tuned, and fully fine-tuned encoders,
- test parameter-efficient adaptation methods.

Expected files:

- `foundation_backbones/`
- `configs/foundation_*.yaml`

## Module 3: Uncertainty Estimation

Purpose:

- estimate prediction confidence,
- generate risk maps,
- evaluate calibration,
- identify failure cases.

Expected methods:

- entropy maps,
- MC dropout,
- deep ensembles,
- evidential segmentation,
- temperature scaling.

## Module 4: Temporal Segmentation

Purpose:

- use video frames instead of independent images,
- reduce flickering,
- improve dynamic-object stability,
- incorporate short-term and long-term memory.

Expected methods:

- feature propagation,
- temporal transformer,
- recurrent memory,
- optical-flow-guided refinement.

## Module 5: Multi-Task Learning

Purpose:

- combine semantic segmentation with geometry and motion,
- learn richer scene representations,
- improve safety-critical classes.

Expected tasks:

- semantic segmentation,
- depth estimation,
- optical flow,
- motion segmentation,
- future segmentation.

## Module 6: Continual Learning

Purpose:

- adapt to new cities, weather, and datasets,
- reduce catastrophic forgetting,
- support long-term deployment.

Expected methods:

- replay memory,
- knowledge distillation,
- adapter tuning,
- LoRA,
- pseudo-labeling.

## Module 7: Explainability and Language Guidance

Purpose:

- explain segmentation predictions,
- support language-conditioned segmentation,
- create human-interpretable failure analysis.

Expected outputs:

- masks,
- confidence maps,
- explanations,
- prompt-conditioned predictions.

## Module 8: Scene Graphs and World Models

Purpose:

- transform pixel predictions into structured driving-scene representations,
- connect perception with prediction and planning,
- represent relations between road, sidewalk, vehicles, pedestrians, motion, and risk.

Expected outputs:

- scene graph nodes,
- scene graph edges,
- risk-aware world state,
- future-scene prediction.

## Development Principle

Each module should be independently testable and should not break the original segmentation baseline. New work should be introduced through configuration files, small reusable components, and clear experiment reports.
