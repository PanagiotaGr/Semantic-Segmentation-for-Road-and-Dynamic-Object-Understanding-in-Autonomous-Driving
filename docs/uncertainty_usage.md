# Uncertainty Utility Usage

This guide shows how to use the uncertainty utilities added under `src/research/uncertainty.py`.

## Why uncertainty matters

Semantic segmentation models usually output a class for every pixel. For autonomous driving, this is not enough. The model should also expose whether a prediction is reliable, especially near pedestrians, vehicles, road boundaries, occlusions, and unusual weather or lighting conditions.

## Basic usage from logits

```python
import torch

from src.research.uncertainty import logits_to_uncertainty

# Example model output: batch=2, classes=5, height=256, width=256
logits = torch.randn(2, 5, 256, 256)

maps = logits_to_uncertainty(logits)

prediction = maps.prediction
confidence = maps.confidence
entropy = maps.entropy
normalized_entropy = maps.normalized_entropy
risk = maps.risk
```

## Outputs

| Output | Shape | Meaning |
| --- | --- | --- |
| `probabilities` | `[B, C, H, W]` | softmax class probabilities |
| `prediction` | `[B, H, W]` | predicted class index |
| `confidence` | `[B, H, W]` | maximum class probability |
| `entropy` | `[B, H, W]` | raw Shannon entropy |
| `normalized_entropy` | `[B, H, W]` | entropy scaled to `[0, 1]` |
| `risk` | `[B, H, W]` | simple uncertainty score, `1 - confidence` |

## Risk thresholding

```python
from src.research.uncertainty import threshold_risk_map

high_risk_pixels = threshold_risk_map(risk, threshold=0.5)
```

## Monte Carlo dropout

If the model uses dropout layers that remain active at inference time, multiple stochastic forward passes can estimate epistemic uncertainty.

```python
import torch

from src.research.uncertainty import mc_dropout_uncertainty

# T stochastic passes, batch=2, classes=5, height=256, width=256
stochastic_logits = torch.randn(8, 2, 5, 256, 256)

mc_maps = mc_dropout_uncertainty(stochastic_logits)

predictive_entropy = mc_maps["predictive_entropy"]
mutual_information = mc_maps["mutual_information"]
risk = mc_maps["risk"]
```

## Research use

Recommended analyses:

- compare uncertainty on correct versus incorrect pixels,
- measure uncertainty around object boundaries,
- inspect uncertainty on pedestrians and vehicles,
- correlate high uncertainty with segmentation errors,
- overlay high-risk regions on the input image,
- evaluate whether uncertainty increases under weather, night, or cross-dataset domain shift.

## Suggested next implementation

The next step is to connect these utilities to an inference script so each predicted mask is saved together with:

- confidence map,
- entropy map,
- risk map,
- high-risk binary mask,
- visual overlay.
