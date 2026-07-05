# Research Roadmap

This roadmap turns the repository from a semantic-segmentation experiment into a PhD-level autonomous-driving perception framework.

## Phase 0: Current Baseline

Current focus:

- CamVid semantic segmentation
- 5-class safety-oriented label space
- U-Net, DeepLabV3+, and SegFormer comparison
- mIoU and class-wise IoU evaluation

Main observed limitation:

- pedestrian segmentation remains weak,
- temporal information is unused,
- uncertainty is not estimated,
- generalization to unseen datasets is not evaluated,
- perception is frame-based rather than world-aware.

## Phase 1: Reproducible Research Baseline

### Goal

Make the current experiments fully reproducible and extensible.

### Tasks

- Standardize train/validation/test split documentation.
- Add random seed control.
- Record hardware and runtime.
- Add experiment configuration files.
- Save metrics in machine-readable CSV or JSON format.
- Add qualitative failure-case gallery.

### Deliverable

A reproducible baseline report and benchmark table.

## Phase 2: Stronger Segmentation Models

### Goal

Improve segmentation quality while maintaining scientific interpretability.

### Candidate Directions

- Higher-resolution input training.
- Better augmentations for small objects.
- Focal loss, Dice loss, Tversky loss, Lovasz loss, and compound losses.
- Transformer tuning with stronger backbones.
- Mask2Former-style decoding.
- Lightweight real-time models for deployment comparison.

### Main Question

Are performance gains caused by architecture, resolution, loss design, label-space design, or data augmentation?

## Phase 3: Foundation-Model Adaptation

### Goal

Use pretrained visual representations to improve robustness and sample efficiency.

### Candidate Approaches

- Frozen encoder plus trainable segmentation decoder.
- Partial fine-tuning.
- Adapter-based fine-tuning.
- LoRA fine-tuning.
- Promptable segmentation refinement.

### Evaluation

- in-domain CamVid performance,
- low-label performance,
- cross-dataset transfer,
- adverse-weather performance.

## Phase 4: Uncertainty-Aware Segmentation

### Goal

Make the model communicate risk, not only class predictions.

### Candidate Approaches

- Monte Carlo dropout.
- Deep ensembles.
- Evidential segmentation.
- Temperature scaling.
- Entropy-based risk maps.

### Evaluation

- expected calibration error,
- negative log-likelihood,
- uncertainty on failure cases,
- uncertainty for pedestrians and vehicles,
- correlation between uncertainty and segmentation error.

## Phase 5: Temporal and 4D Segmentation

### Goal

Use video information to improve consistency and dynamic-object understanding.

### Candidate Approaches

- temporal feature aggregation,
- recurrent segmentation memory,
- optical-flow-guided feature propagation,
- temporal transformer encoder,
- segmentation flicker loss.

### Evaluation

- frame mIoU,
- temporal consistency,
- flicker rate,
- dynamic-object stability,
- video qualitative analysis.

## Phase 6: Multi-Task World-Aware Learning

### Goal

Learn a latent driving-scene representation that contains semantics, geometry, and motion.

### Tasks

- semantic segmentation,
- depth estimation,
- optical flow,
- motion segmentation,
- future segmentation prediction,
- lane or drivable-area estimation.

### Research Question

Does auxiliary geometric and motion supervision improve semantic segmentation of safety-critical classes?

## Phase 7: Continual Learning

### Goal

Allow the model to adapt to new environments without forgetting previous ones.

### Continual Protocols

- city-by-city learning,
- weather-by-weather learning,
- dataset-by-dataset learning,
- day-to-night adaptation.

### Candidate Approaches

- replay memory,
- knowledge distillation,
- regularization-based continual learning,
- adapter isolation,
- LoRA per domain,
- teacher-student pseudo-labeling.

### Metrics

- average accuracy,
- backward transfer,
- forward transfer,
- forgetting score,
- retained performance on previous domains.

## Phase 8: Explainable and Language-Guided Segmentation

### Goal

Make dense perception interpretable and controllable.

### Candidate Outputs

- prediction mask,
- confidence map,
- text explanation,
- failure reason,
- prompt-conditioned segmentation.

### Example Prompt

"Segment road construction objects and vulnerable road users."

Expected model behavior:

- focus on pedestrians,
- detect temporary obstacles,
- mark drivable area uncertainty,
- explain ambiguous regions.

## Phase 9: Scene Graph and World Model

### Goal

Move from pixels to structured scene understanding.

### Scene Graph Nodes

- road,
- sidewalk,
- vehicle,
- pedestrian,
- lane region,
- obstacle,
- uncertain region.

### Scene Graph Edges

- on-road,
- adjacent-to,
- moving-towards,
- occluding,
- crossing,
- high-risk-near.

### Future Extension

Use the scene graph as input to planning, trajectory prediction, or risk assessment.

## Four-Year PhD Timeline

### Year 1

- reproduce baseline,
- expand evaluation,
- add stronger losses and higher resolution,
- write first workshop or conference paper.

### Year 2

- foundation-model adaptation,
- uncertainty-aware segmentation,
- cross-domain evaluation.

### Year 3

- temporal video segmentation,
- multi-task depth/motion/segmentation,
- dynamic-object robustness.

### Year 4

- continual learning,
- scene graph / world model,
- final thesis integration,
- open-source benchmark release.
