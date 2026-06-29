# Contributing

Thank you for your interest in improving this semantic segmentation project.

## How to Contribute

Useful contributions include:

* improving model training scripts,
* adding new segmentation architectures,
* improving evaluation metrics,
* adding visualizations,
* improving documentation,
* adding reproducibility details,
* benchmarking inference speed.

## Development Setup

```bash
pip install -r requirements.txt
```

Before submitting changes, run:

```bash
python -m compileall .
```

## Coding Guidelines

* Keep functions small and readable.
* Use clear variable names.
* Document experiment assumptions.
* Avoid committing large datasets or model checkpoints.
* Store only curated qualitative examples in the repository.

## Scientific Guidelines

When adding a new experiment, document:

* model architecture,
* backbone,
* loss function,
* optimizer,
* learning rate,
* number of epochs,
* random seed,
* mIoU and class-wise IoU,
* qualitative observations.
