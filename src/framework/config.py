from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_TOP_LEVEL = {"experiment", "data", "model", "training", "output"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated YAML-backed experiment configuration."""

    raw: dict[str, Any]
    source: Path

    @property
    def name(self) -> str:
        return str(self.raw["experiment"]["name"])


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")

    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    missing = sorted(_REQUIRED_TOP_LEVEL - payload.keys())
    if missing:
        raise ValueError(f"Missing required configuration sections: {', '.join(missing)}")

    experiment = payload["experiment"]
    if not isinstance(experiment, dict) or not experiment.get("name"):
        raise ValueError("experiment.name must be a non-empty value.")

    return ExperimentConfig(raw=payload, source=source)
