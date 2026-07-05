# Temporal Segmentation Module

This module moves the project from single-frame segmentation toward video-based 4D scene understanding.

## Research Question

Does temporal context improve segmentation stability and dynamic-object understanding in autonomous-driving scenes?

## Motivation

Frame-independent segmentation can flicker across video frames. In autonomous driving, unstable segmentation can cause unreliable downstream planning. Temporal information can improve consistency, motion understanding, and small-object detection.

## Planned Methods

1. **Frame-independent baseline**
   - current segmentation model applied to each frame independently.

2. **Temporal feature fusion**
   - combine features from previous and current frames.

3. **Optical-flow-guided propagation**
   - warp previous predictions or features into the current frame.

4. **Recurrent memory**
   - maintain a hidden scene state over time.

5. **Temporal transformer**
   - use attention across a short video clip.

## Metrics

- frame-level mIoU
- class-wise IoU
- temporal consistency score
- flicker rate
- dynamic-object stability
- boundary consistency
- pedestrian tracking consistency

## Expected Outputs

- per-frame segmentation masks
- temporal consistency plots
- flicker heatmaps
- side-by-side video comparisons
- failure-case clips

## Key Hypothesis

Temporal context should improve dynamic-object segmentation and reduce prediction flicker, especially for pedestrians, vehicles, and ambiguous boundaries.
