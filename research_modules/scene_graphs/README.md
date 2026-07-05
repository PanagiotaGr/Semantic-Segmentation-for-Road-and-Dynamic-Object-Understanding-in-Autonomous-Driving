# Scene Graph and World Model Module

This module converts dense segmentation outputs into structured scene representations for autonomous-driving reasoning.

## Research Question

Can semantic segmentation, depth, motion, and uncertainty be transformed into a structured world state that supports explainable driving perception?

## Motivation

Pixel masks are useful but limited. A driving agent also needs relational knowledge:

- Is the pedestrian on the sidewalk or crossing the road?
- Is a vehicle near the ego lane?
- Is an uncertain region close to the drivable area?
- Is an object moving toward the road?

A scene graph can represent these relations explicitly.

## Candidate Nodes

- road region
- sidewalk region
- pedestrian
- vehicle
- lane or drivable area
- uncertain region
- obstacle
- dynamic object

## Candidate Edges

- `on_road`
- `on_sidewalk`
- `adjacent_to`
- `near_drivable_area`
- `moving_towards`
- `occluding`
- `crossing`
- `high_risk_near`

## Candidate Pipeline

```text
image/video
   |
   v
segmentation + uncertainty + depth + motion
   |
   v
region extraction
   |
   v
object and relation inference
   |
   v
scene graph / world state
```

## Evaluation Ideas

- graph consistency across frames
- relation accuracy if annotations exist
- human interpretability study
- usefulness for downstream risk detection
- correlation between graph risk score and segmentation failures

## Expected Outputs

- region graph JSON
- visualization overlay
- risk-aware scene summary
- temporal scene graph sequence

## Expected Contribution

This module moves the project beyond semantic segmentation toward structured scene understanding and world-model-style perception.
