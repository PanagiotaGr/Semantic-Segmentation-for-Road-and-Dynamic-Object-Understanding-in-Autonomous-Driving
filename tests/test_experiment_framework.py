from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.framework import create_run_directory, load_config, write_run_manifest


def _write_config(path: Path, output_root: Path) -> None:
    payload = {
        "experiment": {"name": "smoke_test", "seed": 7},
        "data": {"dataset": "camvid"},
        "model": {"name": "unet"},
        "training": {"epochs": 1},
        "output": {"root": str(output_root)},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_config_rejects_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("experiment:\n  name: invalid\n", encoding="utf-8")

    try:
        load_config(path)
    except ValueError as exc:
        assert "Missing required configuration sections" in str(exc)
    else:
        raise AssertionError("Invalid configuration was accepted")


def test_run_initialization_writes_contract_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "outputs"
    _write_config(config_path, output_root)

    config = load_config(config_path)
    run_dir = create_run_directory(config)
    manifest = write_run_manifest(config, run_dir)

    assert config.name == "smoke_test"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "predictions").is_dir()
    stored = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored["seed"] == 7
    assert manifest["status"] == "initialized"
