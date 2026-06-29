"""Unified training entrypoint for CamVid semantic segmentation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.camvid_dataset import CamVidSegmentationDataset
from models.model_factory import SUPPORTED_MODELS, create_segmentation_model
from utils.augmentations import get_train_augmentations, get_val_augmentations
from utils.losses import CombinedCrossEntropyDiceLoss, DiceLoss, FocalLoss
from utils.training_tools import EarlyStopping, ExperimentLogger, build_scheduler, set_seed


CLASS_NAMES = ["Background", "Road", "Sidewalk", "Vehicle", "Pedestrian"]


def build_loss(name: str):
    """Create a loss function from a short name."""
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


def compute_batch_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Compute mean IoU for one validation batch."""
    preds = predictions.reshape(-1).detach().cpu().numpy()
    true = targets.reshape(-1).detach().cpu().numpy()
    ious = []
    for class_id in range(num_classes):
        pred_mask = preds == class_id
        true_mask = true == class_id
        union = np.logical_or(pred_mask, true_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_mask, true_mask).sum()
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion, device: str, num_classes: int) -> tuple[float, float]:
    """Run validation and return validation loss and mIoU."""
    model.eval()
    total_loss = 0.0
    miou_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            masks = masks.long().to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item()
            miou_scores.append(compute_batch_iou(preds, masks, num_classes))
    return total_loss / max(len(loader), 1), float(np.mean(miou_scores)) if miou_scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train segmentation models on CamVid.")
    parser.add_argument("--data-dir", default="data/CamVid")
    parser.add_argument("--model", default="unet", choices=SUPPORTED_MODELS)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--loss", default="ce", choices=["ce", "dice", "focal", "ce_dice"])
    parser.add_argument("--scheduler", default="plateau", choices=["none", "plateau", "cosine"])
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--patience", default=7, type=int)
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(CLASS_NAMES)
    experiment_name = args.experiment_name or f"{args.model}_{args.loss}_{args.image_size}_seed{args.seed}"

    data_dir = Path(args.data_dir)
    train_dataset = CamVidSegmentationDataset(data_dir / "train", data_dir / "train_labels", image_size=args.image_size, transform=get_train_augmentations(args.image_size, strong=args.strong_aug))
    val_dataset = CamVidSegmentationDataset(data_dir / "val", data_dir / "val_labels", image_size=args.image_size, transform=get_val_augmentations(args.image_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = create_segmentation_model(args.model, num_classes=num_classes, encoder_name=args.encoder).to(device)
    criterion = build_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(optimizer, args.scheduler)
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
    logger = ExperimentLogger()

    best_miou = 0.0
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{experiment_name}.pth"

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_miou = validate(model, val_loader, criterion, device, num_classes)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("mIoU/val", val_miou, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mIoU={val_miou:.4f}")

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_miou)
            else:
                scheduler.step()

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

        if early_stopping.step(val_miou):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    logger.save(experiment_name, config=vars(args), metrics={"best_miou": best_miou, "checkpoint": str(checkpoint_path)})
    writer.close()


if __name__ == "__main__":
    main()
