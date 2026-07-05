# Uncertainty Implementation Plan

## Goal

Integrate uncertainty-aware analysis into the segmentation pipeline without changing the existing training logic.

## Step 1: Standalone utility

Status: implemented.

- Convert logits to probabilities.
- Compute prediction masks.
- Compute confidence maps.
- Compute entropy maps.
- Compute normalized entropy maps.
- Compute risk maps.
- Support Monte Carlo dropout uncertainty.

## Step 2: Prediction-script integration

Planned.

- Load trained model checkpoint.
- Run inference as before.
- Pass logits into `logits_to_uncertainty`.
- Save uncertainty outputs next to existing prediction masks.

## Step 3: Visualization

Planned.

- Save grayscale confidence maps.
- Save grayscale entropy maps.
- Save high-risk overlays on input images.
- Save side-by-side panels: input, prediction, entropy, risk.

## Step 4: Quantitative analysis

Planned.

- Compute uncertainty on correct pixels.
- Compute uncertainty on incorrect pixels.
- Compute uncertainty near boundaries.
- Compute uncertainty for pedestrians and vehicles.
- Identify high-confidence false negatives.

## Step 5: Research extension

Planned.

- Evaluate uncertainty under cross-dataset transfer.
- Evaluate uncertainty under adverse weather.
- Compare softmax entropy, MC dropout, and ensembles.
- Use uncertainty to select failure cases for manual analysis.
