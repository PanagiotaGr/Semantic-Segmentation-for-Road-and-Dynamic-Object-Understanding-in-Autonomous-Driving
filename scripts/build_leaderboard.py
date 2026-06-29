"""Build a leaderboard from JSON experiment logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_log(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    metrics = payload.get("metrics", {})
    return {
        "experiment_name": payload.get("experiment_name"),
        "created_utc": payload.get("created_utc"),
        "model": config.get("model") or config.get("architecture"),
        "encoder": config.get("encoder"),
        "loss": config.get("loss"),
        "image_size": config.get("image_size"),
        "seed": config.get("seed"),
        "best_miou": metrics.get("best_miou"),
        "checkpoint": metrics.get("checkpoint"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create benchmark leaderboard from experiment logs.")
    parser.add_argument("--log-dir", default=Path("results/experiment_logs"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/leaderboard"), type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_files = sorted(args.log_dir.glob("*.json"))
    if not log_files:
        raise FileNotFoundError(f"No experiment logs found in {args.log_dir}")

    table = pd.DataFrame([flatten_log(path) for path in log_files])
    table = table.sort_values("best_miou", ascending=False, na_position="last")
    table.to_csv(args.output_dir / "leaderboard.csv", index=False)
    (args.output_dir / "leaderboard.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
