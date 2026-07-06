# Temporal Prediction Evaluation

Use `scripts/evaluate_temporal_predictions.py` to compute temporal stability metrics from saved predicted masks.

## Input

A PyTorch tensor file with shape:

```text
[frames, height, width]
```

Each value should be a predicted class index.

## Command

```bash
python scripts/evaluate_temporal_predictions.py \
  --input outputs/video_predictions.pt \
  --output outputs/temporal_report.json \
  --num-classes 5
```

## Output

The script writes a JSON report containing:

- frame change rate,
- consecutive-frame IoU,
- temporal stability,
- input shape,
- number of classes.

## Research use

This script can compare frame-independent segmentation against future temporal models such as feature fusion, recurrent memory, optical-flow-guided refinement, or temporal transformers.
