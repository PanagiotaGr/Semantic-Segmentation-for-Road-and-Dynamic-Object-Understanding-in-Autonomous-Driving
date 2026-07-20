"""Unified configuration-driven semantic-segmentation training entrypoint."""

from __future__ import annotations

import argparse
import json

from src.framework.config import load_experiment_config
from src.framework.data import build_dataloaders
from src.framework.experiment import create_run_directory, set_global_seed, write_run_manifest
from src.framework.models import build_model
from src.framework.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    seed = int(config.raw["experiment"].get("seed", 42))
    deterministic = bool(config.raw["experiment"].get("deterministic", True))
    set_global_seed(seed, deterministic)

    run_dir = create_run_directory(config)
    manifest = write_run_manifest(config, run_dir)
    try:
        train_loader, val_loader = build_dataloaders(config.raw)
        model = build_model(config.raw)
        result = train_model(model, train_loader, val_loader, config.raw, run_dir)
        manifest["status"] = "completed"
        manifest["result"] = {
            "best_validation_miou": result["best_validation_miou"],
            "epochs": result["epochs"],
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
