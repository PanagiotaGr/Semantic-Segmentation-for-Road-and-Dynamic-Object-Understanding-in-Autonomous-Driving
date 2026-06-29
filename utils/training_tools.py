"""Training utilities for semantic segmentation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch


class EarlyStopping:
    """Simple early stopping based on validation metric improvement."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "max") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        """Return True when training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def set_seed(seed: int) -> None:
    """Set random seeds for more reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentLogger:
    """Save experiment metadata and metrics as JSON files."""

    output_dir: str = "results/experiment_logs"

    def save(self, experiment_name: str, config: dict[str, Any], metrics: dict[str, Any]) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        payload = {
            "experiment_name": experiment_name,
            "created_utc": timestamp,
            "config": config,
            "metrics": metrics,
        }
        output_file = path / f"{timestamp}_{experiment_name}.json"
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_file


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str):
    """Build a common learning-rate scheduler."""
    name = scheduler_name.lower()
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {scheduler_name}")
