# Experiment Matrix

This document defines the experimental axes needed to transform the project into a PhD-grade research benchmark.

## Baseline Experiments

| ID | Model | Dataset | Input | Loss | Purpose |
| --- | --- | --- | --- | --- | --- |
| B1 | U-Net ResNet34 | CamVid | 256x256 | Cross-Entropy | current best baseline |
| B2 | U-Net ResNet34 | CamVid | 512x512 | Cross-Entropy | resolution effect |
| B3 | DeepLabV3+ | CamVid | 256x256 | Cross-Entropy | CNN comparison |
| B4 | SegFormer MiT-B0 | CamVid | 256x256 | Cross-Entropy | transformer baseline |
| B5 | SegFormer MiT-B0 | CamVid | 512x512 | Cross-Entropy | transformer resolution effect |

## Loss Function Ablations

| ID | Loss | Target Question |
| --- | --- | --- |
| L1 | Cross-Entropy | reference loss |
| L2 | Weighted Cross-Entropy | class imbalance correction |
| L3 | Focal Loss | rare/difficult class emphasis |
| L4 | Dice Loss | overlap optimization |
| L5 | Cross-Entropy + Dice | hybrid stability and overlap |
| L6 | Tversky Loss | false-negative control |
| L7 | Lovasz Softmax | IoU surrogate optimization |

## Data and Augmentation Ablations

| ID | Intervention | Target Question |
| --- | --- | --- |
| A1 | baseline augmentation | reference |
| A2 | stronger color jitter | robustness to lighting |
| A3 | random scale crop | small-object robustness |
| A4 | weather augmentation | adverse-weather robustness |
| A5 | motion blur / noise | camera degradation robustness |
| A6 | class-balanced crop sampling | pedestrian and vehicle improvement |

## Foundation-Model Adaptation Experiments

| ID | Strategy | Trainable Parameters | Target Question |
| --- | --- | --- | --- |
| F1 | frozen encoder + decoder | low | representation transfer |
| F2 | partial fine-tuning | medium | adaptation-performance trade-off |
| F3 | full fine-tuning | high | upper-bound performance |
| F4 | adapters | low-medium | efficient adaptation |
| F5 | LoRA | low-medium | continual/domain adaptation readiness |

## Uncertainty Experiments

| ID | Method | Output | Target Question |
| --- | --- | --- | --- |
| U1 | softmax entropy | risk map | simple uncertainty baseline |
| U2 | MC dropout | epistemic uncertainty | uncertainty under model ambiguity |
| U3 | deep ensemble | calibrated uncertainty | stronger uncertainty baseline |
| U4 | temperature scaling | calibrated probabilities | post-hoc calibration |
| U5 | evidential segmentation | evidence map | explicit uncertainty modeling |

## Temporal Experiments

| ID | Method | Input | Target Question |
| --- | --- | --- | --- |
| T1 | frame independent | single image | baseline |
| T2 | previous-frame feature fusion | 2 frames | short-term consistency |
| T3 | optical-flow propagation | 2+ frames | motion-aware refinement |
| T4 | recurrent memory | video clip | memory-based stability |
| T5 | temporal transformer | video clip | long-range temporal context |

## Multi-Task Experiments

| ID | Tasks | Target Question |
| --- | --- | --- |
| M1 | segmentation only | baseline |
| M2 | segmentation + depth | geometry helps segmentation? |
| M3 | segmentation + optical flow | motion helps dynamic objects? |
| M4 | segmentation + motion mask | dynamic-object emphasis |
| M5 | segmentation + future segmentation | predictive scene understanding |
| M6 | segmentation + depth + flow + future | world-aware representation |

## Continual-Learning Experiments

| ID | Sequence | Method | Target Question |
| --- | --- | --- | --- |
| C1 | dataset-by-dataset | naive fine-tuning | forgetting baseline |
| C2 | dataset-by-dataset | replay | retention improvement |
| C3 | dataset-by-dataset | distillation | old-model preservation |
| C4 | weather-by-weather | adapters | domain-specialized adaptation |
| C5 | city-by-city | LoRA | parameter-efficient continual learning |

## Explainability and Language Experiments

| ID | Capability | Target Question |
| --- | --- | --- |
| X1 | saliency maps | which regions drive predictions? |
| X2 | uncertainty overlay | where is the model unreliable? |
| X3 | text explanation of failure cases | can failures be summarized? |
| X4 | language-guided segmentation | can prompts change target focus? |
| X5 | scenario retrieval | can language identify safety-critical scenes? |

## Scene Graph Experiments

| ID | Input | Output | Target Question |
| --- | --- | --- | --- |
| G1 | segmentation mask | object-region graph | can masks become structured scenes? |
| G2 | segmentation + depth | spatial graph | does geometry improve relations? |
| G3 | segmentation + flow | motion graph | can dynamic risk be represented? |
| G4 | multi-task output | world-model state | can perception support planning? |

## Reporting Template

Each experiment should report:

- dataset,
- split,
- model,
- input resolution,
- random seed,
- optimizer,
- learning rate,
- loss,
- number of epochs,
- hardware,
- training time,
- inference FPS,
- mIoU,
- class-wise IoU,
- pedestrian IoU,
- vehicle IoU,
- calibration metrics if applicable,
- qualitative examples,
- failure cases.
