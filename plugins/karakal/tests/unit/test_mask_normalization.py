"""One binarization for every mask: exact PNG, JPEG of a black-and-white mask, soft maps, edge cases."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core.mask_normalization import (
    KIND_BINARY,
    KIND_BINARY_LIKE,
    KIND_SOFT,
    METHOD_CONFIG,
    METHOD_EXACT,
    METHOD_FIXED,
    METHOD_FIXED_FALLBACK,
    METHOD_OTSU,
    STATUS_EMPTY,
    STATUS_FULL,
    STATUS_NEAR_EMPTY,
    STATUS_OK,
    MaskNormalizationConfig,
    normalize_mask,
)


def _cell_mask(shape: tuple[int, int] = (400, 520)) -> np.ndarray:
    """Sparse rectangular cells with gaps, like a partly filled array."""

    image = np.zeros(shape, dtype=np.uint8)
    for row in range(5):
        for col in range(9):
            if (row * 9 + col) % 4 == 3:
                continue
            x, y = 20 + col * 54, 24 + row * 72
            cv2.rectangle(image, (x, y), (x + 17, y + 44), 255, -1)
    return image


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def _components(binary: np.ndarray) -> tuple[int, np.ndarray]:
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return count - 1, labels


def test_true_binary_png_is_kept_without_a_threshold() -> None:
    source = _cell_mask()
    result = normalize_mask(source)
    assert result.kind == KIND_BINARY
    assert result.method == METHOD_EXACT
    assert result.status == STATUS_OK
    assert np.array_equal(result.binary, source > 0)


@pytest.mark.parametrize("quality", [60, 75, 90, 95])
def test_jpeg_copy_gives_the_same_components_as_the_lossless_mask(quality: int) -> None:
    source = _cell_mask()
    lossless = normalize_mask(source)
    compressed = _jpeg(source, quality)
    assert int(np.count_nonzero(compressed > 0)) > int(np.count_nonzero(source))  # the trap ``> 0`` falls into

    result = normalize_mask(compressed)
    assert result.kind == KIND_BINARY_LIKE
    assert result.method == METHOD_FIXED
    expected_count, expected_labels = _components(lossless.binary)
    count, labels = _components(result.binary)
    assert count == expected_count
    for label in range(1, expected_count + 1):
        cell = expected_labels == label
        partner = np.bincount(labels[cell]).argmax()
        assert partner > 0
        other = labels == partner
        assert np.count_nonzero(cell & other) / np.count_nonzero(cell | other) >= 0.93


def test_empty_jpeg_frame_with_noise_is_empty_not_specks() -> None:
    # A real empty mask (TEST2_05121) keeps JPEG noise up to gray 53; Otsu splits it into 3 objects.
    rng = np.random.default_rng(3)
    image = np.zeros((300, 300), dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(5, 290, size=2)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(1, 3)), int(rng.integers(15, 54)), -1)
    result = normalize_mask(_jpeg(image, 85))
    assert result.status == STATUS_EMPTY
    assert not result.binary.any()


def test_black_and_white_frames_are_empty_and_full() -> None:
    assert normalize_mask(np.zeros((40, 40), dtype=np.uint8)).status == STATUS_EMPTY
    full = normalize_mask(np.full((40, 40), 255, dtype=np.uint8))
    assert full.status == STATUS_FULL
    assert full.binary.all()


def test_a_single_tiny_object_is_kept_and_reported_near_empty() -> None:
    image = np.zeros((1200, 1200), dtype=np.uint8)
    image[600:603, 600:603] = 255
    result = normalize_mask(_jpeg(image, 90))
    assert result.status == STATUS_NEAR_EMPTY
    assert 1 <= int(np.count_nonzero(result.binary)) <= 16


def _soft_blobs() -> np.ndarray:
    """Probability-like map: smooth blobs with wide gray slopes."""

    yy, xx = np.mgrid[0:200, 0:200].astype(np.float64)
    field = np.zeros((200, 200))
    for cx, cy in ((50, 50), (140, 60), (90, 150)):
        field = np.maximum(field, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 22.0**2)))
    return field


def test_soft_two_mode_map_uses_otsu_and_records_it() -> None:
    field = _soft_blobs()
    sharpened = np.clip((field - 0.35) * 3.0, 0.0, 1.0)
    result = normalize_mask(sharpened)
    assert result.kind == KIND_SOFT
    assert result.method == METHOD_OTSU
    assert 0.0 < result.threshold < 255.0
    assert result.metadata()["method"] == METHOD_OTSU


def test_soft_map_without_two_modes_falls_back_to_the_fixed_middle() -> None:
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.float64), (64, 1)).astype(np.uint8)
    result = normalize_mask(ramp)
    assert result.kind == KIND_SOFT
    assert result.method == METHOD_FIXED_FALLBACK
    assert result.threshold == 128.0
    assert np.array_equal(result.binary, ramp >= 128)


def test_configured_threshold_wins_in_levels_and_fractions() -> None:
    field = _soft_blobs()
    by_fraction = normalize_mask(field, MaskNormalizationConfig(threshold=0.7))
    by_level = normalize_mask((field * 255).astype(np.uint8), MaskNormalizationConfig(threshold=178.5))
    assert by_fraction.method == METHOD_CONFIG
    assert by_fraction.threshold == pytest.approx(178.5)
    assert abs(int(np.count_nonzero(by_fraction.binary)) - int(np.count_nonzero(by_level.binary))) <= 40


def test_inversion_happens_only_when_asked() -> None:
    source = 255 - _cell_mask()
    plain = normalize_mask(source)
    inverted = normalize_mask(source, MaskNormalizationConfig(invert=True))
    assert plain.inverted is False and inverted.inverted is True
    assert np.array_equal(inverted.binary, _cell_mask() > 0)
    assert np.array_equal(plain.binary, ~inverted.binary)


def test_color_and_bool_inputs() -> None:
    source = _cell_mask()
    color = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    assert np.array_equal(normalize_mask(color).binary, source > 0)
    assert np.array_equal(normalize_mask(source > 0).binary, source > 0)
