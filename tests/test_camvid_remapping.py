from __future__ import annotations

import numpy as np

from src.framework.labels import CAMVID_12_TO_5, mapping_from_config, remap_mask
from src.framework.visualization import colorize_mask


def test_camvid_12_to_5_mapping() -> None:
    mask = np.arange(12, dtype=np.uint8).reshape(3, 4)
    remapped = remap_mask(mask, CAMVID_12_TO_5)
    assert remapped.tolist() == [[0, 0, 0, 1], [2, 0, 0, 0], [3, 4, 3, 0]]


def test_unknown_labels_use_ignore_index() -> None:
    mask = np.array([[0, 12]], dtype=np.uint8)
    remapped = remap_mask(mask, CAMVID_12_TO_5, ignore_index=255)
    assert remapped.tolist() == [[0, 255]]


def test_dense_mapping_from_config() -> None:
    assert mapping_from_config([0, 1, 1]) == {0: 0, 1: 1, 2: 1}


def test_colorized_mask_is_rgb() -> None:
    image = colorize_mask(np.array([[0, 1], [3, 4]], dtype=np.uint8))
    assert image.mode == "RGB"
    assert image.size == (2, 2)
