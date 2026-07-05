# Multi-Task World-Aware Learning Module

This module extends semantic segmentation into richer scene understanding by jointly learning semantics, geometry, motion, and prediction.

## Research Question

Does learning depth, motion, and future scene structure improve safety-critical semantic segmentation for autonomous driving?

## Motivation

A segmentation model that only predicts pixel labels may fail to understand physical scene structure. A world-aware model should learn what objects are present, where they are, how they move, and how the scene may evolve.

## Candidate Tasks

1. **Semantic segmentation**
   - road, sidewalk, vehicles, pedestrians, background.

2. **Depth estimation**
   - approximate scene geometry from monocular images.

3. **Optical flow**
   - pixel-level motion between frames.

4. **Motion segmentation**
   - static versus dynamic regions.

5. **Future segmentation**
   - prediction of the semantic layout at a future time step.

6. **Drivable-area or lane-region estimation**
   - safety-oriented navigation support.

## Architecture Concept

```text
input image or video clip
        |
        v
shared encoder
        |
        +--> semantic segmentation head
        +--> depth head
        +--> optical flow head
        +--> motion head
        +--> future segmentation head
```

## Metrics

- semantic mIoU
- class-wise IoU
- pedestrian IoU
- vehicle IoU
- depth error if depth labels are available
- flow endpoint error if flow labels are available
- future segmentation mIoU
- multi-task transfer gain

## Main Ablations

- segmentation only
- segmentation + depth
- segmentation + optical flow
- segmentation + motion
- segmentation + future prediction
- all tasks combined

## Expected Contribution

The module should test whether auxiliary geometry and motion supervision help the model build a stronger latent scene representation than segmentation alone.
