from __future__ import annotations

import numpy as np

from karakal.core.source_mask_from_gray import (
    SOURCE_MASK_ALGO_VERSION,
    SourceMaskConfig,
    build_source_mask,
    mask_dice,
    mask_iou,
)


def _bright_rectangle(size: int = 96, *, inset: int = 24, bright: int = 220, dark: int = 30) -> tuple[np.ndarray, np.ndarray]:
    gray = np.full((size, size), dark, dtype=np.uint8)
    gray[inset : size - inset, inset : size - inset] = bright
    mask = np.zeros((size, size), dtype=bool)
    mask[inset : size - inset, inset : size - inset] = True
    return gray, mask


def test_build_source_mask_recovers_bright_rectangle() -> None:
    gray, expected = _bright_rectangle()
    predicted = build_source_mask(gray, config=SourceMaskConfig(min_region_area=40))
    assert predicted.shape == expected.shape
    assert mask_iou(predicted, expected) >= 0.55
    assert SOURCE_MASK_ALGO_VERSION == "v1"


def test_mask_iou_and_dice_helpers() -> None:
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[2:6, 2:6] = True
    b[2:6, 2:6] = True
    assert mask_iou(a, b) == 1.0
    assert mask_dice(a, b) == 1.0
    b[:, :] = False
    b[3:7, 3:7] = True
    assert 0.0 < mask_iou(a, b) < 1.0
    assert 0.0 < mask_dice(a, b) < 1.0


def test_build_source_mask_resizes_to_target_shape() -> None:
    gray, _expected = _bright_rectangle(size=64, inset=16)
    predicted = build_source_mask(gray, target_shape=(32, 32), config=SourceMaskConfig(min_region_area=10))
    assert predicted.shape == (32, 32)
    assert predicted.dtype == bool


def test_flat_grayscale_returns_empty_mask() -> None:
    gray = np.full((48, 48), 128, dtype=np.uint8)
    predicted = build_source_mask(gray)
    assert predicted.shape == gray.shape
    assert not np.any(predicted)
