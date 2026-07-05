# Uncertainty Utilities PR Summary

This change adds the first implementation component from the PhD research roadmap: uncertainty-aware segmentation analysis.

## Added files

- `src/research/uncertainty.py`
- `tests/test_uncertainty.py`
- `docs/uncertainty_usage.md`
- `docs/uncertainty_research_note.md`
- `configs/uncertainty_analysis.yaml`
- `.github/workflows/research-tests.yml`

## Main capability

The new utility converts segmentation logits or probabilities into:

- predicted masks,
- confidence maps,
- entropy maps,
- normalized entropy maps,
- risk maps,
- high-risk binary masks,
- Monte Carlo dropout uncertainty estimates.

## Scientific value

This makes the project start moving beyond accuracy-only semantic segmentation toward safety-aware perception, where the model can identify uncertain or risky regions.
