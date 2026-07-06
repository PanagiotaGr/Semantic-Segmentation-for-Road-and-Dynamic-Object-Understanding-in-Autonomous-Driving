# Temporal Fusion Baseline

This module introduces a minimal temporal feature-fusion baseline.

## Motivation

Before using complex temporal transformers or recurrent memory, the project needs a simple baseline that can combine information from multiple frames.

## Added modules

`src/research/temporal_fusion.py` provides:

- `TemporalFeatureFusion`
- `TemporalFusionSegmentationHead`

## Expected feature shape

```text
[batch, frames, channels, height, width]
```

## Fusion modes

- `mean`: average features over time,
- `sum`: sum features over time,
- `concat`: concatenate frame features along the channel dimension.

## Example

```python
import torch
from src.research.temporal_fusion import TemporalFeatureFusion, TemporalFusionSegmentationHead

features = torch.randn(2, 3, 64, 32, 32)
fusion = TemporalFeatureFusion(mode="mean")
fused = fusion(features)

head = TemporalFusionSegmentationHead(input_channels=64, num_classes=5)
logits = head(fused)
```

## Research use

Use this as the first temporal model baseline before adding optical-flow guidance, recurrent memory, or temporal transformers.
