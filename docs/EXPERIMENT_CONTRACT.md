# Experiment Reproducibility Contract

Every reported result in this repository must be traceable to a complete experiment record.

## Required metadata

Each run must record:

- experiment identifier;
- date and time;
- Git commit SHA;
- dataset name and version;
- train, validation, and test split definition;
- class mapping and ignored labels;
- input resolution;
- normalization and augmentation;
- model architecture and encoder;
- pretrained-weight source;
- loss function and parameters;
- optimizer and scheduler;
- learning rate;
- batch size;
- number of epochs;
- random seed;
- hardware;
- software versions;
- best-checkpoint selection rule;
- final checkpoint path.

## Required outputs

A completed experiment should produce:

```text
outputs/<experiment_id>/
├── config.yaml
├── environment.json
├── metrics.json
├── class_metrics.csv
├── training_history.csv
├── best_checkpoint.pt
├── predictions/
├── failure_cases/
└── experiment_report.md
```

## Required evaluation

At minimum, report:

- mean Intersection over Union;
- class-wise IoU;
- Dice or F1 score;
- class-wise precision and recall;
- confusion matrix;
- critical-class results for pedestrians and vehicles.

Where uncertainty is evaluated, also report:

- Expected Calibration Error;
- Brier score or negative log-likelihood;
- reliability diagram;
- error-detection AUROC;
- risk-coverage curve.

Where temporal methods are evaluated, also report:

- consecutive-frame IoU;
- frame-change rate;
- temporal stability;
- critical-class temporal consistency.

## Claim policy

A result may be called an improvement only when:

1. it uses the same data split and evaluation protocol as the baseline;
2. all changed variables are documented;
3. the metric difference is reported exactly;
4. negative or neutral findings are retained;
5. no future or incomplete experiment is described as completed.

## Repetition policy

Important experiments should be repeated with multiple seeds when computationally feasible. Mean and variability must be reported separately from the best run.

## Thesis use

Only results satisfying this contract should be treated as final thesis evidence. Preliminary results may be discussed only when clearly labelled as preliminary.
