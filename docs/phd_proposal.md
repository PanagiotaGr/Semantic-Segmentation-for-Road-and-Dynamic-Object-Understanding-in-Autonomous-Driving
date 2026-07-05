# PhD Research Proposal

## Working Title

**Towards World-Aware and Continually Adaptive Semantic Scene Understanding for Autonomous Driving**

## Motivation

The current repository establishes a semantic-segmentation baseline for autonomous-driving perception using a safety-oriented reduced label space. The next research step is to move beyond frame-level pixel classification and develop a perception system that reasons about structure, motion, uncertainty, temporal continuity, and domain shift.

Autonomous-driving perception systems must not only identify road, sidewalk, vehicles, and pedestrians. They must also estimate whether a prediction is reliable, remain stable across video frames, adapt to unseen cities and weather conditions, and support downstream reasoning for planning and risk assessment.

## Central Research Question

How can semantic segmentation be extended into a world-aware, uncertainty-aware, temporally consistent, and continually adaptive scene-understanding framework for autonomous driving?

## Hypothesis

A multi-task perception architecture that jointly learns semantic segmentation, depth, motion, temporal consistency, uncertainty, and scene-level structure will generalize better than a single-task segmentation model under challenging autonomous-driving conditions.

## Research Objectives

1. Establish strong and reproducible semantic-segmentation baselines.
2. Improve generalization using modern vision foundation backbones and domain-robust training.
3. Add uncertainty estimation for safety-critical perception.
4. Add temporal memory to reduce video flickering and improve dynamic-object segmentation.
5. Add multi-task learning for segmentation, depth, optical flow, and future scene prediction.
6. Add continual-learning mechanisms to adapt across cities, weather, and datasets without catastrophic forgetting.
7. Represent driving scenes as structured scene graphs or world-model states for downstream reasoning.
8. Explore language-guided and explainable segmentation for human-interpretable autonomous perception.

## Proposed Contributions

### Contribution 1: Robust Segmentation Baseline

Create a standardized benchmark across U-Net, DeepLabV3+, SegFormer, Mask2Former-style decoders, and foundation-model encoders. Evaluate the effect of resolution, augmentation, loss design, class grouping, and model capacity.

### Contribution 2: Foundation-Model Adaptation

Investigate whether frozen or partially fine-tuned visual foundation models improve robustness under domain shift. Candidate encoders include self-supervised Vision Transformers and segmentation-oriented backbones.

### Contribution 3: Uncertainty-Aware Perception

Extend segmentation outputs from class predictions to class-risk estimates. Evaluate calibration, entropy, expected calibration error, and failure detection on rare and safety-critical classes.

### Contribution 4: Temporal and 4D Scene Understanding

Use video sequences to enforce temporal consistency and improve small dynamic-object segmentation. Candidate modules include temporal transformers, recurrent memory, feature propagation, and temporal consistency losses.

### Contribution 5: Multi-Task World-Aware Learning

Jointly train segmentation with auxiliary tasks such as monocular depth estimation, optical flow, motion segmentation, and future segmentation. The goal is to learn a richer latent scene representation than pixel labels alone.

### Contribution 6: Continual Learning Across Domains

Develop a continual semantic-segmentation protocol where the model is sequentially exposed to different cities, datasets, weather types, or lighting conditions. Evaluate performance retention and forward transfer using replay, distillation, adapter tuning, or LoRA.

### Contribution 7: Explainable and Language-Guided Segmentation

Connect dense predictions with textual explanations and language prompts. This can support debugging, scenario search, and human-in-the-loop safety analysis.

### Contribution 8: Scene Graph and World Model

Convert segmentation, motion, depth, and object cues into structured scene representations. A frame or video clip can be represented as a graph containing road, sidewalk, vehicles, pedestrians, traffic context, movement, and risk.

## Research Methodology

The project will follow an incremental experimental design:

1. Reproduce the current CamVid baseline.
2. Build a unified training and evaluation protocol.
3. Add one research capability at a time.
4. Evaluate each contribution under controlled ablations.
5. Compare in-domain and cross-domain performance.
6. Analyze failure modes qualitatively and quantitatively.

## Datasets

Initial dataset:

- CamVid

Recommended extensions:

- Cityscapes
- BDD100K
- ACDC
- Mapillary Vistas
- WildDash
- KITTI / KITTI-360

## Evaluation Metrics

Core segmentation metrics:

- mIoU
- class-wise IoU
- pixel accuracy
- boundary F1 score

Safety-critical metrics:

- pedestrian IoU
- vehicle IoU
- false negative rate for dynamic objects
- calibration error
- uncertainty-risk correlation

Robustness metrics:

- cross-dataset mIoU
- adverse-weather mIoU
- night/day performance gap
- temporal flicker rate
- continual-learning forgetting score

## Expected Thesis Structure

1. Introduction and problem formulation
2. Semantic road-scene segmentation baseline
3. Robust foundation-model adaptation
4. Uncertainty-aware segmentation
5. Temporal and multi-task scene understanding
6. Continual learning for autonomous-driving perception
7. Scene graphs, world models, and explainability
8. Conclusions and future work

## Publication Strategy

Potential paper sequence:

1. Robust baseline and label-space study for safety-oriented road segmentation
2. Foundation-model adaptation for cross-domain autonomous-driving segmentation
3. Uncertainty-aware dynamic-object segmentation under domain shift
4. Temporal multi-task segmentation for video-based road-scene understanding
5. Continual semantic segmentation across cities and weather conditions
6. Scene-graph or world-model representation for explainable driving perception

## Success Criteria

The project is successful if it produces:

- a reproducible benchmark,
- stronger dynamic-object segmentation,
- improved robustness under domain shift,
- calibrated uncertainty estimates,
- temporally stable video predictions,
- a continual-learning protocol,
- and a structured world-aware representation beyond pixel labels.
