"""Shared training loop and segmentation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid] + prediction[valid]
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, Any]:
    matrix = matrix.double()
    intersection = matrix.diag()
    union = matrix.sum(1) + matrix.sum(0) - intersection
    iou = torch.where(union > 0, intersection / union, torch.nan)
    return {
        "miou": float(torch.nanmean(iou).item()),
        "class_iou": [None if torch.isnan(value) else float(value.item()) for value in iou],
        "pixel_accuracy": float(intersection.sum().div(matrix.sum().clamp_min(1)).item()),
    }


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_cfg = config["training"]["optimizer"]
    name = str(optimizer_cfg["name"]).lower()
    lr = float(optimizer_cfg["learning_rate"])
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            predictions = logits.argmax(dim=1)
            matrix += confusion_matrix(predictions.cpu(), targets.cpu(), num_classes)

    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    num_classes = int(config["data"]["num_classes"])
    history: list[dict[str, Any]] = []
    best_miou = float("-inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, num_classes, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, num_classes)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": record}, run_dir / "checkpoints" / "last.pt")
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": record}, run_dir / "checkpoints" / "best.pt")

    return {"best_validation_miou": best_miou, "epochs": epochs, "history": history}
