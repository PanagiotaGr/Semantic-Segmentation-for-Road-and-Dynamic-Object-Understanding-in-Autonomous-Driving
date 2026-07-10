# Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-green)
![Dataset](https://img.shields.io/badge/Dataset-CamVid-orange)
![Research](https://img.shields.io/badge/Research-Autonomous%20Driving-purple)

A research-oriented semantic segmentation project for urban road-scene understanding, with emphasis on safety-critical perception for autonomous driving.

The repository combines a completed CamVid baseline study with reusable research utilities for uncertainty estimation, temporal consistency, temporal feature fusion, and continual-learning experiments.

## Research Objective

The central question is:

> How do architecture choice, label-space design, loss weighting, uncertainty estimation, and temporal information affect the reliable segmentation of static road structures and dynamic road users?

The project focuses particularly on road, sidewalk, vehicle, and pedestrian perception. These classes are important because segmentation errors may propagate to downstream modules such as localization, trajectory prediction, planning, and risk assessment.

## Current Project Status

### Implemented baseline study

- CamVid semantic segmentation pipeline.
- Comparison of U-Net, DeepLabV3+, and SegFormer.
- Original 12-class and reduced 5-class label formulations.
- Standard and weighted cross-entropy experiments.
- Quantitative evaluation using mIoU and class-wise IoU.
- Qualitative prediction masks, overlays, and comparison plots.

### Implemented research utilities

- Pixel-wise confidence, entropy, normalized entropy, and risk maps.
- Monte Carlo dropout uncertainty utilities.
- Temporal stability metrics for saved prediction sequences.
- Temporal difference maps and change-frequency analysis.
- Temporal consistency losses.
- Sliding-window temporal datasets.
- Mean, sum, and concatenation-based temporal feature fusion.
- Replay memory for continual-learning experiments.
- Unit tests for the reusable research modules.

### Research directions under development

- Full integration of uncertainty outputs into inference scripts.
- End-to-end temporal model training.
- Cross-dataset and adverse-weather evaluation.
- Continual semantic segmentation across datasets and domains.
- Foundation-model adaptation and parameter-efficient fine-tuning.
- Multi-task learning, scene graphs, and world-aware perception.

## Main Contributions

1. A controlled comparison of convolutional and transformer-based segmentation architectures.
2. A safety-oriented reduction of the CamVid label space from 12 classes to 5 classes.
3. An ablation study of standard versus weighted cross-entropy loss.
4. Class-specific evaluation of vehicles and pedestrians rather than reliance on global mIoU alone.
5. Reusable uncertainty and temporal-analysis modules for extending frame-based segmentation.
6. A modular research roadmap toward robust, adaptive, and temporally consistent autonomous-driving perception.

## Dataset

The baseline experiments use the **CamVid dataset**, a road-scene semantic segmentation benchmark containing urban driving images and pixel-level annotations.

### Reduced five-class label space

The original CamVid labels are grouped into five higher-level categories:

| Reduced class | Interpretation |
| --- | --- |
| Background | Non-critical or merged scene elements |
| Road | Drivable road surface |
| Sidewalk | Pedestrian-side navigable area |
| Vehicle | Cars and other road vehicles |
| Pedestrian | Human road users |

The reduced formulation is intended to improve interpretability and focus evaluation on classes with direct relevance to autonomous-driving perception.

## Models

### CNN-based architectures

- **U-Net with ResNet34 encoder**
- **DeepLabV3+**

### Transformer-based architecture

- **SegFormer with MiT-B0 backbone**

## Baseline Experimental Setup

| Component | Configuration |
| --- | --- |
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
| --- | --- | --- | ---: |
| U-Net | 12 classes | Cross-Entropy | 0.4797 |
| U-Net | 5 classes | Cross-Entropy | **0.7240** |
| DeepLabV3+ | 5 classes | Cross-Entropy | 0.7132 |
| U-Net | 5 classes | Weighted Cross-Entropy | 0.6823 |
| SegFormer | 5 classes | Cross-Entropy | 0.6135 |

### Class-wise IoU for the best U-Net model

| Class | IoU |
| --- | ---: |
| Background | 0.9646 |
| Road | 0.9591 |
| Sidewalk | 0.8424 |
| Vehicle | 0.5944 |
| Pedestrian | 0.2596 |

## Interpretation of Results

### Label-space reduction

Reducing the original 12-class task to a five-class safety-oriented formulation produced a substantial increase in mIoU. This result indicates that label-space design is not merely a preprocessing choice; it directly changes the complexity and practical interpretation of the learning problem.

### Weighted loss

Weighted cross-entropy did not outperform standard cross-entropy in the reported U-Net experiment. Class weighting may increase attention to rare classes, but it can also reduce stability or degrade performance on dominant classes. The result therefore motivates further comparison with focal, Dice, Tversky, and hybrid losses.

### CNNs and transformers

The CNN-based models outperformed the tested SegFormer configuration. This finding should not be interpreted as a general limitation of transformers. It more likely reflects the small dataset, reduced image resolution, optimization settings, and limited architecture-specific tuning.

### Safety-critical classes

Road and sidewalk achieved high IoU because they occupy large, spatially continuous regions. Vehicles and especially pedestrians were more difficult because they occupy fewer pixels and exhibit greater variation in scale, pose, occlusion, and appearance.

A high global mIoU is therefore insufficient by itself. For autonomous-driving applications, pedestrian and vehicle performance must be examined separately.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving.git
cd Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving

python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> PyTorch installation depends on the available CPU or CUDA environment. The dependency file includes the CUDA 12.1 package index, but users should select the appropriate PyTorch installation for their hardware when necessary.

## Data Preparation

The CamVid dataset is not distributed through this repository. Place the images and masks in a local `data/` directory and adapt the paths used by the experiment scripts.

A reproducible experiment should document:

- the exact train, validation, and test split,
- the original-to-reduced class mapping,
- image resizing and normalization,
- data augmentation,
- ignored labels,
- random seed,
- checkpoint and output locations.

## Running the Baseline Experiments

The baseline workflow is:

1. Prepare the CamVid images and masks.
2. Train the selected segmentation architecture.
3. Generate predictions for the test split.
4. Compute mIoU and class-wise IoU.
5. Save masks, overlays, plots, and failure cases.

Training and inference files use model-specific names. Inspect the repository root for the available `train_*.py` and `predict_*.py` scripts before running an experiment.

Typical commands follow this pattern:

```bash
python train_<model>.py
python predict_<model>.py
```

## Research Utilities

### Uncertainty estimation

The module `src/research/uncertainty.py` supports:

- softmax probabilities,
- predicted class masks,
- confidence maps,
- entropy and normalized entropy maps,
- risk maps,
- high-risk masks,
- Monte Carlo dropout uncertainty.

Example:

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

See `docs/uncertainty_usage.md` for additional details.

### Temporal evaluation

Saved prediction sequences with shape `[frames, height, width]` can be evaluated using:

```bash
python scripts/evaluate_temporal_predictions.py \
  --input outputs/video_predictions.pt \
  --output outputs/temporal_report.json \
  --num-classes 5
```

The report includes:

- frame change rate,
- consecutive-frame IoU,
- temporal stability.

Temporal difference maps can be generated with:

```bash
python scripts/generate_temporal_difference_maps.py \
  --input outputs/video_predictions.pt \
  --output-dir outputs/temporal_difference
```

### Temporal modeling components

The repository includes reusable components for:

- sliding-window sequence construction,
- temporal segmentation datasets,
- probability- and logit-consistency losses,
- mean, sum, and concatenation-based feature fusion,
- a minimal temporal segmentation head.

The reference configuration is available at `configs/temporal_fusion_baseline.yaml`.

### Continual learning

The replay-memory utility in `src/research/replay_memory.py` stores examples from earlier domains so that they can be mixed with new-domain data during sequential training.

```python
from src.research.replay_memory import ReplayMemory

memory = ReplayMemory(capacity=100, seed=42)
memory.extend(camvid_samples, domain="camvid")
replay_batch = memory.sample(batch_size=8)
```

This provides a basic continual-learning baseline before introducing distillation, adapters, LoRA, or domain-specific memory policies.

## Testing

Install `pytest` if it is not already available:

```bash
pip install pytest
```

Run the complete test suite:

```bash
pytest
```

The tests cover uncertainty estimation, temporal metrics, temporal difference maps, temporal losses, temporal datasets, temporal feature fusion, replay memory, and continual-learning metrics where available.

## Qualitative Results

### Segmentation examples

![Reduced prediction example](sample_outputs/reduced_prediction_1.png)
![Weighted overlay example](sample_outputs/weighted_overlay_1.png)

### Performance plots

![mIoU comparison](plots/miou_comparison_full.png)
![Class IoU comparison](plots/class_iou_full.png)

## Repository Structure

```text
.
├── configs/                  # Experiment and research configuration files
├── docs/                     # Research notes, protocols, and usage guides
├── plots/                    # Quantitative comparison plots
├── research_modules/         # Modular research roadmap and module documentation
├── sample_outputs/           # Example predictions and overlays
├── scripts/                  # Evaluation and analysis scripts
├── src/
│   └── research/             # Reusable uncertainty, temporal, and continual-learning utilities
├── templates/                # Experiment reporting templates
├── tests/                    # Unit tests for research utilities
├── train_*.py                # Model-specific training scripts
├── predict_*.py              # Model-specific inference scripts
├── requirements.txt
└── README.md
```

## Reproducibility Checklist

A reported experiment should include:

- dataset version and split,
- class mapping,
- input resolution,
- preprocessing and augmentation,
- model architecture and pretrained weights,
- optimizer and learning-rate schedule,
- loss function and class weights,
- batch size and number of epochs,
- random seed,
- hardware and software environment,
- checkpoint identifier,
- mIoU and class-wise IoU,
- qualitative successes and failures.

The repository distinguishes between completed baseline results and proposed research extensions. Proposed modules should not be interpreted as experimentally validated until corresponding training configurations, checkpoints, and result files are reported.

## Limitations

- The baseline experiments use 256 × 256 inputs, which limits small-object detail.
- CamVid is relatively small compared with modern autonomous-driving datasets.
- Pedestrian segmentation remains weak because of class imbalance and small spatial extent.
- SegFormer was not exhaustively tuned.
- The reported baseline does not yet evaluate cross-dataset generalization.
- Temporal utilities are available, but a complete end-to-end temporal training study is still required.
- Real-time latency, memory usage, and deployment performance have not yet been benchmarked.

## Future Work

- Higher-resolution and class-balanced training.
- Focal, Dice, Tversky, Lovász, and combined losses.
- Stronger transformer and foundation-model backbones.
- Cross-dataset evaluation on Cityscapes, BDD100K, ACDC, and related datasets.
- Uncertainty calibration and error-detection analysis.
- End-to-end temporal segmentation with motion-aware alignment.
- Continual learning across cities, datasets, weather, and lighting conditions.
- Multi-task learning with depth, optical flow, and future segmentation.
- Risk-aware scene graphs and structured world representations.

The broader research plan is documented in:

- `docs/phd_proposal.md`
- `docs/research_roadmap.md`
- `docs/experiment_matrix.md`
- `docs/ACADEMIC_COMPLETION.md`

## Author

**Panagiota Grosdouli**

## License

This repository is currently intended for academic and research use. No formal open-source license has been declared yet. Until a license is added, reuse and redistribution rights are not automatically granted.

## Citation

```bibtex
@software{Grosdouli_Semantic_Segmentation_2026,
  author = {Grosdouli, Panagiota},
  title = {Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving},
  year = {2026},
  url = {https://github.com/PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving}
}
```
