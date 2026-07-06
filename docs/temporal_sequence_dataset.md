# Temporal Sequence Dataset

The temporal dataset utilities create sliding windows over ordered frame paths.

## Why this matters

Temporal segmentation models need input sequences instead of isolated frames. A sequence dataset is the first step toward feature fusion, recurrent memory, optical-flow guidance, and temporal transformers.

## Utilities

`src/research/temporal_dataset.py` provides:

- `TemporalSequenceSample`
- `build_sliding_windows`
- `TemporalSegmentationDataset`

## Example

```python
from pathlib import Path
from src.research.temporal_dataset import build_sliding_windows

image_paths = sorted(Path("data/frames").glob("*.png"))
mask_paths = sorted(Path("data/masks").glob("*.png"))

samples = build_sliding_windows(
    image_paths=image_paths,
    mask_paths=mask_paths,
    sequence_length=3,
    stride=1,
)
```

## Expected tensor shapes

When used with transforms that return image tensors as `[channels, height, width]`, the dataset returns:

```text
images: [frames, channels, height, width]
masks:  [frames, height, width]
```

## Research use

Use this dataset helper to train temporal baselines and compare them against frame-independent segmentation.
