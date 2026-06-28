"""Experiment configuration definitions for the segmentation project."""

from __future__ import annotations

from dataclasses import dataclass, asdict


REDUCED_CLASS_NAMES: list[str] = [
    "Background",
    "Road",
    "Sidewalk",
    "Vehicle",
    "Pedestrian",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration object used to document one experiment."""

    experiment_name: str
    model_name: str
    backbone: str
    num_classes: int = 5
    image_size: int = 256
    batch_size: int = 4
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    loss_function: str = "CrossEntropyLoss"
    epochs: int = 25
    random_seed: int = 42
    dataset_name: str = "CamVid"

    def to_dict(self) -> dict[str, object]:
        """Return a serializable dictionary."""
        return asdict(self)


def default_experiments() -> list[ExperimentConfig]:
    """Return the main configurations described in the project."""
    return [
        ExperimentConfig("unet_5class_ce", "U-Net", "ResNet34"),
        ExperimentConfig("unet_5class_weighted_ce", "U-Net", "ResNet34", loss_function="WeightedCrossEntropyLoss"),
        ExperimentConfig("deeplabv3plus_5class_ce", "DeepLabV3+", "encoder-dependent"),
        ExperimentConfig("segformer_mitb0_5class_ce", "SegFormer", "MiT-B0", optimizer="AdamW", learning_rate=5e-5),
    ]
