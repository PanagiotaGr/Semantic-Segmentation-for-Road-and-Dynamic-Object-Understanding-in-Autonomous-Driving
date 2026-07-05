# Uncertainty Utilities PR Summary

This change documents the first implementation component from the PhD research roadmap: uncertainty-aware segmentation analysis.

## Existing implementation

The project now includes `src/research/uncertainty.py`, which converts segmentation logits or probabilities into:

- predicted masks,
- confidence maps,
- entropy maps,
- normalized entropy maps,
- risk maps,
- high-risk binary masks,
- Monte Carlo dropout uncertainty estimates.

## Added in this documentation update

- module README,
- staged implementation plan,
- PR-level summary for reviewers.

## Scientific value

This moves the project beyond accuracy-only semantic segmentation toward safety-aware perception, where the model can identify uncertain or risky regions.
