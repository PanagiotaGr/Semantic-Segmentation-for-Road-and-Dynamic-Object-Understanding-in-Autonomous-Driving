# Uncertainty Module

This module extends semantic segmentation from hard class prediction to risk-aware perception.

## Research Question

Can the segmentation model identify when its own predictions are unreliable, especially for safety-critical classes such as pedestrians and vehicles?

## Motivation

For autonomous driving, a wrong prediction with high confidence is dangerous. A useful perception system should expose uncertainty so downstream planning can treat ambiguous regions with caution.

## Planned Methods

1. **Softmax entropy**
   - simple baseline uncertainty map,
   - no retraining required.

2. **Monte Carlo dropout**
   - multiple stochastic forward passes,
   - estimates epistemic uncertainty.

3. **Deep ensembles**
   - multiple independently trained models,
   - stronger but more expensive uncertainty baseline.

4. **Temperature scaling**
   - post-hoc calibration,
   - improves probability reliability.

5. **Evidential segmentation**
   - predicts evidence rather than only probability,
   - explicitly models uncertainty.

## Metrics

- Expected Calibration Error
- Negative Log-Likelihood
- Brier Score
- uncertainty-error correlation
- mean uncertainty on correct pixels
- mean uncertainty on incorrect pixels
- uncertainty near object boundaries
- uncertainty for pedestrians and vehicles

## Expected Outputs

- segmentation mask
- confidence map
- entropy map
- error map
- uncertainty overlay
- calibration report

## Safety-Critical Analysis

The most important cases are:

- missed pedestrians,
- confused vehicle/background regions,
- uncertain road boundaries,
- night or adverse-weather failures,
- occluded dynamic objects.
