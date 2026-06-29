# Prediction and Leaderboard Pipeline

## Unified Prediction

Use `predict.py` to generate masks, color masks, and optional overlays from a trained checkpoint.

```bash
python predict.py \
  --model unet \
  --encoder resnet34 \
  --checkpoint models/best_unet_ce_256_seed42.pth \
  --image-dir data/CamVid/test \
  --output-dir results/predictions/unet \
  --save-overlays
```

Outputs:

```text
results/predictions/unet/
├── masks/
├── color_masks/
└── overlays/
```

## Evaluation

After generating prediction masks, evaluate them with:

```bash
python scripts/evaluate_predictions.py \
  --gt-dir data/CamVid/test_labels_reduced \
  --pred-dir results/predictions/unet/masks \
  --output-dir results/evaluation/unet
```

This creates:

* class metrics CSV,
* raw confusion matrix CSV,
* normalized confusion matrix CSV,
* confusion matrix heatmap PNG.

## Leaderboard

After training multiple experiments, build a leaderboard from experiment logs:

```bash
python scripts/build_leaderboard.py
```

Outputs:

```text
results/leaderboard/
├── leaderboard.csv
└── leaderboard.md
```

## Why This Matters

The prediction and leaderboard pipeline makes the repository easier to evaluate scientifically. It supports consistent comparison between models and keeps results traceable through saved checkpoints, logs, metrics, and visual outputs.
