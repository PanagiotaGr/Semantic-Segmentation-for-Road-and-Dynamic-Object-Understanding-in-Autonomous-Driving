from pathlib import Path

from src.research.temporal_dataset import build_sliding_windows


def test_build_sliding_windows_count():
    image_paths = [Path(f"frame_{index}.png") for index in range(5)]
    samples = build_sliding_windows(image_paths, sequence_length=3, stride=1)
    assert len(samples) == 3


def test_build_sliding_windows_with_stride():
    image_paths = [Path(f"frame_{index}.png") for index in range(6)]
    samples = build_sliding_windows(image_paths, sequence_length=3, stride=2)
    assert len(samples) == 2
    assert samples[1].image_paths[0] == Path("frame_2.png")


def test_build_sliding_windows_with_masks():
    image_paths = [Path(f"frame_{index}.png") for index in range(4)]
    mask_paths = [Path(f"mask_{index}.png") for index in range(4)]
    samples = build_sliding_windows(image_paths, mask_paths, sequence_length=2)
    assert len(samples) == 3
    assert samples[0].mask_paths is not None
    assert samples[0].mask_paths[1] == Path("mask_1.png")
