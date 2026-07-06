# Temporal Reporting

Temporal experiments should report both segmentation accuracy and temporal stability.

## Template

Use:

```text
templates/temporal_experiment_report.md
```

## Why this matters

A temporal model can improve stability while reducing frame-level accuracy, or improve mIoU while still flickering. Reporting both kinds of metrics prevents misleading conclusions.

## Required comparison

Every temporal experiment should compare against a frame-independent baseline using:

- mIoU,
- class-wise IoU,
- vehicle IoU,
- pedestrian IoU,
- frame change rate,
- consecutive-frame IoU,
- temporal stability.

## Recommended qualitative outputs

- predicted mask sequence,
- temporal difference maps,
- change-frequency map,
- failure-case examples,
- video overlay if available.
