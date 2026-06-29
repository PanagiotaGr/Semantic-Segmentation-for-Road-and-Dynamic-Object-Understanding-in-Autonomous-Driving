"""Model factory for semantic segmentation architectures.

Supported segmentation_models_pytorch architectures:
U-Net, DeepLabV3+, PSPNet, FPN, PAN, LinkNet, MAnet, and UPerNet when available.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp


def create_segmentation_model(
    architecture: str,
    num_classes: int = 5,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
):
    """Create a segmentation model by architecture name."""
    name = architecture.lower().replace("-", "").replace("_", "")

    common_kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": num_classes,
    }

    if name == "unet":
        return smp.Unet(**common_kwargs)
    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common_kwargs)
    if name == "deeplabv3":
        return smp.DeepLabV3(**common_kwargs)
    if name == "pspnet":
        return smp.PSPNet(**common_kwargs)
    if name == "fpn":
        return smp.FPN(**common_kwargs)
    if name == "pan":
        return smp.PAN(**common_kwargs)
    if name == "linknet":
        return smp.Linknet(**common_kwargs)
    if name == "manet":
        return smp.MAnet(**common_kwargs)
    if name == "upernet":
        if not hasattr(smp, "UPerNet"):
            raise ValueError("UPerNet is not available in the installed segmentation_models_pytorch version")
        return smp.UPerNet(**common_kwargs)

    raise ValueError(f"Unsupported architecture: {architecture}")


SUPPORTED_MODELS = [
    "unet",
    "deeplabv3plus",
    "deeplabv3",
    "pspnet",
    "fpn",
    "pan",
    "linknet",
    "manet",
    "upernet",
]
