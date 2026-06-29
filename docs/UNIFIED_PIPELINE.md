# Unified Training Pipeline

The repository now includes a unified training entrypoint:

```bash
python train.py --model unet --loss ce --image-size 256 --epochs 25
```

## Supported CNN Models

| Command value | Architecture |
| --- | --- |
| `unet` | U-Net |
| `deeplabv3plus` | DeepLabV3+ |
| `deeplabv3` | DeepLabV3 |
| `pspnet` | PSPNet |
| `fpn` | Feature Pyramid Network |
| `pan` | Pyramid Attention Network |
| `linknet` | LinkNet |
| `manet` | MAnet |

## Supported Loss Functions

| Command value | Loss |
| --- | --- |
| `ce` | Cross-Entropy |
| `dice` | Dice Loss |
| `focal` | Focal Loss |
| `ce_dice` | Combined Cross-Entropy and Dice Loss |

## Example Commands

Train U-Net:

```bash
python train.py --model unet --encoder resnet34 --loss ce --epochs 25
```

Train PSPNet:

```bash
python train.py --model pspnet --encoder resnet34 --loss ce --epochs 25
```

Train FPN with Dice loss:

```bash
python train.py --model fpn --encoder resnet34 --loss dice --epochs 25
```

Run all main CNN models:

```bash
bash scripts/run_full_benchmark.sh
```

## What the Pipeline Saves

The pipeline saves:

* best model checkpoint in `models/`,
* TensorBoard logs in `runs/`,
* experiment JSON logs in `results/experiment_logs/`.

## Why This Matters

This structure makes the project more reproducible and scientifically fair, because all models can be trained with the same dataset split, same image resolution, same loss function, and same training protocol.
