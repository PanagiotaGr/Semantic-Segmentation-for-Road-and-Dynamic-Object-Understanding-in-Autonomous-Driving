from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .config import ExperimentConfig


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_directory(config: ExperimentConfig) -> Path:
    root = Path(config.raw["output"].get("root", "outputs"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / config.name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("checkpoints", "predictions", "reports", "logs"):
        (run_dir / child).mkdir()
    return run_dir


def write_run_manifest(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    seed = int(config.raw["experiment"].get("seed", 42))
    manifest: dict[str, Any] = {
        "experiment_name": config.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config.source),
        "git_commit": _git_commit(),
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "status": "initialized",
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
