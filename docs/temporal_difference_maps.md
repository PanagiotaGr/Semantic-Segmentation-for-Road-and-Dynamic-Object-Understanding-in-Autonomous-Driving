# Temporal Difference Maps

Temporal difference maps show where predicted labels change between consecutive frames.

## Why this matters

A segmentation model can have good frame-level metrics but still flicker over time. Difference maps localize the unstable pixels and make temporal failures easier to inspect.

## Utility functions

`src/research/temporal_diff.py` provides:

- `temporal_change_maps`: binary change maps between consecutive frames,
- `temporal_change_count`: number of label changes per pixel,
- `temporal_change_frequency`: normalized per-pixel change frequency.

## Input shape

```text
[frames, height, width]
```

## Example

```python
import torch
from src.research.temporal_diff import temporal_change_frequency

predictions = torch.randint(0, 5, (5, 256, 256))
frequency = temporal_change_frequency(predictions)
```

## Research use

Use these maps to identify flickering regions, unstable object boundaries, and temporally inconsistent dynamic-object predictions.
