from __future__ import annotations

import argparse

from src.framework import create_run_directory, load_config, set_global_seed, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a reproducible experiment run.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.raw["experiment"].get("seed", 42))
    deterministic = bool(config.raw["experiment"].get("deterministic", True))
    set_global_seed(seed, deterministic=deterministic)

    run_dir = create_run_directory(config)
    write_run_manifest(config, run_dir)
    print(f"Initialized experiment '{config.name}' at {run_dir}")


if __name__ == "__main__":
    main()
