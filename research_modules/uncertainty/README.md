# Uncertainty Module

This module adds uncertainty-aware analysis for semantic segmentation.

## Research question

Can a segmentation model identify when its own dense predictions are unreliable?

## Implemented utilities

The current implementation lives in `src/research/uncertainty.py` and supports:

- logits-to-probability conversion,
- predicted class masks,
- confidence maps,
- entropy maps,
- normalized entropy maps,
- risk maps,
- high-risk masks,
- Monte Carlo dropout uncertainty.

## Tests

Unit tests are available in `tests/test_uncertainty.py`.

## Usage

See `docs/uncertainty_usage.md`.

## Next integration step

Connect the utility to the existing prediction scripts so each inference run can optionally save confidence, entropy, and risk maps alongside segmentation outputs.
