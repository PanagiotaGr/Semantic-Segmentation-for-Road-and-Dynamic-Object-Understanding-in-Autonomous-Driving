"""Generic training template for all supported segmentation models.

This template is designed for experiments across U-Net, DeepLabV3+, PSPNet,
FPN, PAN, LinkNet, MAnet, and other models exposed by models/model_factory.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from models.model_factory import create_segmentation_model, SUPPORTED_MODELS
from utils.losses import DiceLoss, FocalLoss, CombinedCrossEntropyDiceLoss
from utils.training_tools import EarlyStopping, ExperimentLogger, build_scheduler, set_seed


def build_loss(name: str):
    loss_name = name.lower()
    if loss_name == "ce":
        return torch.nn.CrossEntropyLoss()
    if loss_name == "dice":
        return DiceLoss()
    if loss_name == "focal":
        return FocalLoss()
    if loss_name == "ce_dice":
        return CombinedCrossEntropyDiceLoss()
    raise ValueError(f"Unsupported loss: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic segmentation training template.")
    parser.add_argument("--architecture", default="unet", choices=SUPPORTED_MODELS)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--num-classes", default=5, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--loss", default="ce", choices=["ce", "dice", "focal", "ce_dice"])
    parser.add_argument("--scheduler", default="plateau", choices=["none", "plateau", "cosine"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--patience", default=7, type=int)
    parser.add_argument("--experiment-name", default="segmentation_experiment")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        num_classes=args.num_classes,
    ).to(device)

    criterion = build_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, args.scheduler)
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    writer = SummaryWriter(log_dir=f"runs/{args.experiment_name}")
    logger = ExperimentLogger()

    print("This is a reusable training template.")
    print("Connect your DataLoader objects and training loop where indicated.")
    print(f"Model: {args.architecture} | Encoder: {args.encoder} | Loss: {args.loss}")

    best_miou = 0.0
    for epoch in range(args.epochs):
        train_loss = 0.0
        val_miou = 0.0

        # TODO: connect project-specific train_loader and val_loader here.
        # Keep these logging calls when integrating into the existing training scripts.
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("mIoU/val", val_miou, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_miou)
            else:
                scheduler.step()

        if val_miou > best_miou:
            best_miou = val_miou
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"models/best_{args.architecture}_{args.loss}.pth")

        if early_stopping.step(val_miou):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    logger.save(
        experiment_name=args.experiment_name,
        config=vars(args),
        metrics={"best_miou": best_miou},
    )
    writer.close()


if __name__ == "__main__":
    main()
