# Uncertainty Research Note

## Research question

Can a semantic segmentation system identify when its own dense predictions are unreliable?

## Why this matters

For autonomous driving, an incorrect prediction is especially dangerous when the model is confident. Uncertainty maps can help identify ambiguous or high-risk regions before downstream planning uses the segmentation output.

## Initial implementation

The first implementation provides model-agnostic utilities for:

- confidence maps,
- entropy maps,
- normalized entropy,
- risk maps,
- Monte Carlo dropout uncertainty,
- high-risk threshold masks.

## How to use in experiments

For each predicted segmentation mask, save:

- predicted class mask,
- confidence map,
- entropy map,
- normalized entropy map,
- risk map,
- high-risk binary mask.

## Suggested evaluation

1. Compare mean uncertainty on correctly classified pixels versus incorrectly classified pixels.
2. Measure uncertainty near semantic boundaries.
3. Analyze uncertainty for pedestrians and vehicles separately.
4. Test whether uncertainty increases under night, rain, fog, snow, or cross-dataset transfer.
5. Use uncertainty maps to select failure cases for manual inspection.

## Next step

Connect `src/research/uncertainty.py` to the existing prediction scripts so qualitative outputs include uncertainty overlays.
