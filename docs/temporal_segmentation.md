# Temporal Segmentation

This document introduces the first temporal-evaluation layer for video semantic segmentation.

## Motivation

Autonomous-driving perception should be stable across time. A model may achieve good frame-level mIoU while still flickering across consecutive frames. Temporal metrics expose this failure mode.

## Added metrics

The file `src/research/temporal_metrics.py` adds:

- `frame_change_rate`: fraction of pixels whose predicted label changes between consecutive frames,
- `consecutive_frame_iou`: mean IoU between consecutive predicted masks,
- `temporal_stability_score`: compact summary containing change rate, consecutive-frame IoU, and stability.

## Expected prediction shape

Temporal metrics expect predicted labels with shape:

```text
[frames, height, width]
```

## Example

```python
import torch
from src.research.temporal_metrics import temporal_stability_score

predictions = torch.randint(0, 5, (4, 256, 256))
score = temporal_stability_score(predictions, num_classes=5)
```

## Research use

Use these metrics to compare:

- single-frame segmentation,
- temporal feature fusion,
- optical-flow-guided consistency,
- recurrent memory,
- temporal transformers.

## Next step

Connect these metrics to prediction outputs from consecutive CamVid frames and report temporal stability alongside mIoU.
