# Academic Completion Criteria

This document defines the review standard for the autonomous-driving semantic segmentation project.

## Research-grade objective

The project should present a controlled semantic-segmentation study for road-scene understanding, with emphasis on autonomous-driving perception, safety-relevant class grouping, and architecture comparison.

## Minimum completion standard

A professor-level reviewer should be able to verify:

1. The CamVid dataset preparation procedure is documented.
2. The original-to-reduced class mapping is explicit and justified.
3. CNN and transformer baselines are trained and evaluated under comparable settings.
4. Metrics include mIoU and per-class IoU, not only visual examples.
5. Safety-critical classes such as vehicles and pedestrians are discussed separately.
6. Loss weighting and class imbalance are treated experimentally.
7. Qualitative overlays are included for both successful and failed predictions.
8. Training, prediction, and evaluation scripts are documented.
9. Results are connected to exact configuration choices.
10. Limitations are stated, especially reduced resolution, dataset scale, and pedestrian-class difficulty.

## Evaluation protocol

| Component | Evidence expected |
|---|---|
| Dataset | split details, class mapping, preprocessing assumptions |
| Training | model configs, optimizer, learning rate, batch size |
| Quantitative results | mIoU and per-class IoU tables |
| Ablation | original versus reduced label space, weighted versus unweighted loss |
| Qualitative analysis | overlays, masks, failure cases |
| Safety interpretation | discussion of road, sidewalk, vehicle, and pedestrian performance |

## Definition of done

The project is review-ready when the reported results can be regenerated from documented commands and the repository clearly distinguishes executed experiments from proposed future improvements.

## Research interpretation

The strongest final discussion should explain why label-space design affects autonomous-driving perception. High global mIoU is not sufficient if safety-critical classes remain weak; pedestrian and vehicle performance should be interpreted as downstream risk signals for a robotic system.
