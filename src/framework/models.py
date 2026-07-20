"""Model factory for unified semantic-segmentation experiments."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a segmentation model from the validated experiment config."""
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()

    if name in {"unet", "deeplabv3plus"}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "segmentation-models-pytorch is required for CNN models"
            ) from exc

        common = {
            "encoder_name": model_cfg.get("encoder", "resnet34"),
            "encoder_weights": model_cfg.get("encoder_weights"),
            "in_channels": int(model_cfg.get("in_channels", 3)),
            "classes": int(model_cfg["num_classes"]),
        }
        if name == "unet":
            return smp.Unet(**common)
        return smp.DeepLabV3Plus(**common)

    raise ValueError(f"Unsupported model: {name}")
