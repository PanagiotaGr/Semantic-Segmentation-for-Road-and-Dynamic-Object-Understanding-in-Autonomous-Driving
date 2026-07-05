# Uncertainty Artifact Generation

The script `scripts/generate_uncertainty_artifacts.py` creates uncertainty outputs from saved segmentation logits or probabilities.

## Input

A PyTorch tensor file with shape `[classes, height, width]` or `[batch, classes, height, width]`.

## Command

`python scripts/generate_uncertainty_artifacts.py --input outputs/sample_logits.pt --output-dir outputs/uncertainty --input-type logits --risk-threshold 0.5`

## Outputs

For each sample, the script saves prediction, confidence, entropy, normalized entropy, risk, and high-risk mask tensors. It also saves a `metadata.json` file with summary statistics.

## Research use

Use these outputs to identify uncertain regions, compare uncertainty against segmentation errors, and inspect high-risk pedestrian or vehicle predictions.
