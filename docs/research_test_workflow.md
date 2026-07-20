# Research Utility Test Workflow

The workflow `.github/workflows/research-utilities.yml` runs the research utility test suite on pull requests.

## Purpose

The repository now includes several research-oriented modules for uncertainty and temporal segmentation. A unified workflow makes sure these utilities remain importable and testable as the project grows.

## Current tests

The workflow runs tests for:

- uncertainty maps,
- temporal metrics,
- temporal difference maps,
- temporal consistency losses,
- temporal sequence datasets,
- temporal feature fusion.

## Dependencies

The workflow installs:

- PyTorch,
- pytest,
- Pillow.

## Research value

This gives the project a lightweight CI safety net for PhD-level research utilities without requiring full model training in GitHub Actions.
