"""Dataset helpers for temporal semantic segmentation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TemporalSequenceSample:
    """Paths belonging to one temporal training sample."""

    image_paths: Tuple[Path, ...]
    mask_paths: Optional[Tuple[Path, ...]] = None


def build_sliding_windows(
    image_paths: Sequence[Path],
    mask_paths: Optional[Sequence[Path]] = None,
    sequence_length: int = 3,
    stride: int = 1,
) -> List[TemporalSequenceSample]:
    """Build temporal windows from ordered image and optional mask paths."""

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if mask_paths is not None and len(image_paths) != len(mask_paths):
        raise ValueError("image_paths and mask_paths must have the same length")

    samples: List[TemporalSequenceSample] = []
    max_start = len(image_paths) - sequence_length
    for start in range(0, max_start + 1, stride):
        end = start + sequence_length
        image_window = tuple(Path(path) for path in image_paths[start:end])
        mask_window = None
        if mask_paths is not None:
            mask_window = tuple(Path(path) for path in mask_paths[start:end])
        samples.append(TemporalSequenceSample(image_window, mask_window))
    return samples


class TemporalSegmentationDataset(Dataset):
    """Simple temporal dataset returning frame windows and optional masks.

    Returned image tensor shape is [frames, channels, height, width] when
    `image_transform` returns [channels, height, width]. Returned mask tensor
    shape is [frames, height, width] when masks are provided.
    """

    def __init__(
        self,
        samples: Sequence[TemporalSequenceSample],
        image_transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        self.samples = list(samples)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        images = [self.image_transform(Image.open(path).convert("RGB")) for path in sample.image_paths]
        image_tensor = torch.stack(images, dim=0)

        if sample.mask_paths is None:
            return {"images": image_tensor, "image_paths": sample.image_paths}

        if self.mask_transform is None:
            raise ValueError("mask_transform is required when mask paths are provided")

        masks = [self.mask_transform(Image.open(path)) for path in sample.mask_paths]
        mask_tensor = torch.stack(masks, dim=0)
        return {
            "images": image_tensor,
            "masks": mask_tensor,
            "image_paths": sample.image_paths,
            "mask_paths": sample.mask_paths,
        }


__all__ = [
    "TemporalSequenceSample",
    "build_sliding_windows",
    "TemporalSegmentationDataset",
]
