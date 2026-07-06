# Temporal Difference Generation

Use `scripts/generate_temporal_difference_maps.py` to generate temporal flicker maps from saved predicted masks.

## Input

A PyTorch tensor file with shape:

```text
[frames, height, width]
```

Each value should be a predicted class index.

## Command

```bash
python scripts/generate_temporal_difference_maps.py \
  --input outputs/video_predictions.pt \
  --output-dir outputs/temporal_difference
```

## Outputs

The script writes:

- `change_maps.pt`
- `change_count.pt`
- `change_frequency.pt`
- `temporal_difference_summary.json`

## Research use

Use the outputs to locate flickering pixels, unstable object boundaries, and temporally inconsistent predictions for vehicles and pedestrians.
