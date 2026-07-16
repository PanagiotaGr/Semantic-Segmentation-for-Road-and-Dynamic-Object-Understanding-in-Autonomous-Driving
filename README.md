<div align="center">

# Semantic Segmentation for Road and Dynamic Object Understanding

### Safety-oriented urban-scene perception for autonomous driving

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CamVid-F59E0B)](#dataset)
[![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-16A34A)](#research-objective)
[![Status](https://img.shields.io/badge/Status-Active%20Research-6D28D9)](#project-status)

[English](README.md) · [Ελληνικά](README_GR.md)

</div>

<p align="center">
  <img src="sample_outputs/reduced_prediction_1.png" alt="Semantic-segmentation prediction" width="48%">
  <img src="sample_outputs/weighted_overlay_1.png" alt="Semantic-segmentation overlay" width="48%">
</p>

---

## Overview

This repository investigates reliable **semantic segmentation of urban road scenes** for autonomous-driving perception. It combines a completed CamVid baseline study with reusable modules for uncertainty estimation, temporal consistency, temporal feature fusion, and continual-learning experiments.

The work focuses on safety-relevant scene elements—especially **road, sidewalk, vehicles, and pedestrians**—because segmentation failures in these classes can propagate to localization, trajectory prediction, planning, and risk assessment.

## Research Objective

The central research question is:

> How do architecture choice, label-space design, loss weighting, uncertainty estimation, and temporal information affect the reliable segmentation of static road structures and dynamic road users?

The project evaluates three complementary dimensions:

1. **Segmentation quality:** global and class-wise accuracy.
2. **Reliability:** confidence, entropy, uncertainty, and failure detection.
3. **Temporal robustness:** consistency across consecutive driving frames.

## Project Status

### Completed baseline study

- CamVid semantic-segmentation pipeline.
- Comparison of U-Net, DeepLabV3+, and SegFormer.
- Original 12-class and reduced 5-class formulations.
- Standard and weighted cross-entropy experiments.
- Evaluation using mIoU and class-wise IoU.
- Prediction masks, overlays, and comparison plots.

### Implemented research utilities

- Pixel-wise confidence, entropy, normalized entropy, and risk maps.
- Monte Carlo dropout uncertainty estimation.
- Temporal stability metrics and difference maps.
- Temporal consistency losses.
- Sliding-window temporal datasets.
- Mean, sum, and concatenation-based temporal fusion.
- Replay memory for continual-learning experiments.
- Unit tests for reusable research components.

### Under development

- End-to-end uncertainty-aware inference.
- Temporal model training and motion-aware alignment.
- Cross-dataset and adverse-weather evaluation.
- Continual segmentation across domains.
- Foundation-model adaptation and parameter-efficient fine-tuning.

## Main Contributions

1. Controlled comparison of CNN- and transformer-based segmentation architectures.
2. Safety-oriented reduction of the CamVid label space from 12 to 5 classes.
3. Ablation of standard and weighted cross-entropy losses.
4. Class-specific analysis of vehicles and pedestrians instead of relying only on global mIoU.
5. Reusable uncertainty and temporal-analysis modules.
6. A modular roadmap toward robust and adaptive autonomous-driving perception.

## Dataset

The baseline experiments use **CamVid**, an urban road-scene dataset with pixel-level semantic annotations.

### Reduced five-class label space

| Reduced class | Interpretation |
|---|---|
| Background | Non-critical or merged scene elements |
| Road | Drivable road surface |
| Sidewalk | Pedestrian-side navigable area |
| Vehicle | Cars and other road vehicles |
| Pedestrian | Human road users |

The reduced formulation prioritizes interpretability and safety-critical classes while lowering the complexity of the original 12-class problem.

## Models

| Family | Architecture |
|---|---|
| CNN | U-Net with ResNet34 encoder |
| CNN | DeepLabV3+ |
| Transformer | SegFormer with MiT-B0 backbone |

## Experimental Setup

| Component | Configuration |
|---|---|
| Input resolution | 256 × 256 |
| Batch size | 4 |
| CNN optimizer | Adam |
| Transformer optimizer | AdamW |
| CNN learning rate | 1e-3 |
| Transformer learning rate | 5e-5 |
| Loss functions | Cross-Entropy, Weighted Cross-Entropy |
| Primary metric | Mean Intersection over Union |

## Results

### Overall performance

| Model | Label setup | Loss | mIoU |
|---|---|---|---:|
| U-Net | 12 classes | Cross-Entropy | 0.4797 |
| **U-Net** | **5 classes** | **Cross-Entropy** | **0.7240** |
| DeepLabV3+ | 5 classes | Cross-Entropy | 0.7132 |
| U-Net | 5 classes | Weighted Cross-Entropy | 0.6823 |
| SegFormer | 5 classes | Cross-Entropy | 0.6135 |

### Class-wise IoU of the best U-Net model

| Class | IoU |
|---|---:|
| Background | 0.9646 |
| Road | 0.9591 |
| Sidewalk | 0.8424 |
| Vehicle | 0.5944 |
| Pedestrian | 0.2596 |

<p align="center">
  <img src="plots/miou_comparison_full.png" alt="mIoU model comparison" width="48%">
  <img src="plots/class_iou_full.png" alt="Class-wise IoU comparison" width="48%">
</p>

### Interpretation

The reduced five-class formulation produced the strongest overall result, showing that label-space design directly affects both optimization difficulty and task interpretation. Standard cross-entropy outperformed the tested weighted variant, while the CNN models performed better than the current SegFormer configuration.

These results do **not** establish a general superiority of CNNs over transformers. CamVid is small, the input resolution is limited, and the transformer configuration has not been exhaustively tuned.

The low pedestrian IoU highlights an important safety limitation: global mIoU can remain high even when small and rare road users are segmented poorly.

## Installation

```bash
git clone https://github.com/PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving.git
cd Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PyTorch installation depends on the available CPU or CUDA environment. Select the appropriate PyTorch build for the target hardware when necessary.

## Data Preparation

CamVid is not redistributed through this repository. Place images and masks in a local `data/` directory and configure the paths used by the experiment scripts.

A reproducible experiment should record the exact split, class mapping, resizing, normalization, augmentation, ignored labels, random seed, checkpoint, and output paths.

## Running the Baselines

Training and inference scripts follow model-specific names:

```bash
python train_<model>.py
python predict_<model>.py
```

The standard workflow is:

```text
CamVid images and masks
          ↓
Preprocessing and label remapping
          ↓
Model training
          ↓
Pixel-wise prediction
          ↓
mIoU and class-wise IoU
          ↓
Masks, overlays, plots, and failure analysis
```

## Research Utilities

### Uncertainty estimation

```python
import torch
from src.research.uncertainty import logits_to_uncertainty

logits = torch.randn(2, 5, 256, 256)
maps = logits_to_uncertainty(logits)

prediction = maps.prediction
confidence = maps.confidence
entropy = maps.entropy
risk = maps.risk
```

The module supports confidence maps, entropy, normalized entropy, risk maps, high-risk masks, and Monte Carlo dropout uncertainty. Additional guidance is available in `docs/uncertainty_usage.md`.

### Temporal evaluation

```bash
python scripts/evaluate_temporal_predictions.py \
  --input outputs/video_predictions.pt \
  --output outputs/temporal_report.json \
  --num-classes 5
```

```bash
python scripts/generate_temporal_difference_maps.py \
  --input outputs/video_predictions.pt \
  --output-dir outputs/temporal_difference
```

The temporal report includes frame-change rate, consecutive-frame IoU, and temporal stability.

### Continual learning

```python
from src.research.replay_memory import ReplayMemory

memory = ReplayMemory(capacity=100, seed=42)
memory.extend(camvid_samples, domain="camvid")
replay_batch = memory.sample(batch_size=8)
```

This provides a basic replay baseline before introducing distillation, adapters, LoRA, or domain-specific memory strategies.

## Testing

```bash
pip install pytest
pytest
```

Tests cover uncertainty estimation, temporal metrics, difference maps, temporal losses, sequence datasets, feature fusion, replay memory, and continual-learning utilities where available.

## Repository Structure

```text
.
├── configs/                  # Experiment configurations
├── docs/                     # Research notes and usage guides
├── plots/                    # Quantitative comparison plots
├── research_modules/         # Research-roadmap documentation
├── sample_outputs/           # Prediction masks and overlays
├── scripts/                  # Evaluation and analysis scripts
├── src/research/             # Uncertainty, temporal, and continual-learning modules
├── templates/                # Experiment-reporting templates
├── tests/                    # Unit tests
├── train_*.py                # Model-specific training scripts
├── predict_*.py              # Model-specific inference scripts
├── requirements.txt
├── README.md
└── README_GR.md
```

## Reproducibility Checklist

- Dataset version and split.
- Original-to-reduced class mapping.
- Input resolution and preprocessing.
- Augmentation and ignored labels.
- Model architecture and pretrained weights.
- Optimizer, schedule, loss, and class weights.
- Batch size, epochs, and random seed.
- Hardware and software environment.
- Checkpoint identifier.
- mIoU, class-wise IoU, and qualitative failure cases.

The repository distinguishes completed experiments from proposed extensions. A module should not be considered experimentally validated until its configuration, checkpoint, and results are reported.

## Limitations

- The 256 × 256 resolution limits small-object detail.
- CamVid is small compared with modern driving datasets.
- Pedestrian segmentation remains weak.
- SegFormer has not been exhaustively tuned.
- Cross-dataset generalization has not yet been evaluated.
- Temporal utilities exist, but full end-to-end temporal training remains pending.
- Real-time latency, memory consumption, and deployment performance are not yet benchmarked.

## Roadmap

- [x] CamVid preprocessing and baseline training.
- [x] U-Net, DeepLabV3+, and SegFormer comparison.
- [x] Five-class safety-oriented label mapping.
- [x] Standard and weighted-loss experiments.
- [x] Uncertainty and temporal-analysis utilities.
- [x] Continual-learning replay memory.
- [ ] Improve pedestrian and small-object segmentation.
- [ ] Integrate uncertainty into inference and evaluation.
- [ ] Train an end-to-end temporal segmentation model.
- [ ] Evaluate on Cityscapes, BDD100K, and ACDC.
- [ ] Study adverse weather and domain shift.
- [ ] Benchmark latency, memory, and deployment performance.

## Future Work

Future extensions include focal, Dice, Tversky, Lovász, and hybrid losses; stronger transformer or foundation-model backbones; uncertainty calibration; motion-aware temporal alignment; continual learning across cities and weather; and multi-task prediction of depth, optical flow, and future segmentation.

The broader research plan is documented in `docs/phd_proposal.md`, `docs/research_roadmap.md`, `docs/experiment_matrix.md`, and `docs/ACADEMIC_COMPLETION.md`.

## Author

**Panagiota Grosdouli**

## Citation

```bibtex
@software{Grosdouli_Semantic_Segmentation_2026,
  author = {Grosdouli, Panagiota},
  title = {Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving},
  year = {2026},
  url = {https://github.com/PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving}
}
```

## License

No formal open-source license has been declared yet. Until a license is added, reuse and redistribution rights are not automatically granted.
