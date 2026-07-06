# Temporal Consistency Loss

Temporal consistency loss is the first training-side step toward reducing segmentation flicker.

## Motivation

Frame-independent segmentation can produce unstable predictions across consecutive frames. A temporal consistency loss penalizes unnecessary changes between adjacent predictions.

## Added utilities

`src/research/temporal_losses.py` provides:

- `temporal_probability_consistency_loss`
- `temporal_logit_consistency_loss`
- `combined_segmentation_temporal_loss`

## Expected logits shape

```text
[batch, frames, classes, height, width]
```

## Example

```python
import torch
from src.research.temporal_losses import combined_segmentation_temporal_loss

segmentation_loss = torch.tensor(1.0)
logits = torch.randn(2, 3, 5, 256, 256)
loss = combined_segmentation_temporal_loss(segmentation_loss, logits, temporal_weight=0.1)
```

## Research use

Use this as a baseline temporal training objective before adding optical-flow alignment, recurrent memory, or temporal transformers.
