from __future__ import annotations

import cv2
import numpy as np

from contour.vision.metal_recovery.seeded_segmentation import seeded_segmentation_mask
from contour.vision.metal_recovery.segmentation import (
    normalize_metal_segmentation_strategy,
)
from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig
from contour.vision.metal_recovery.structural_watershed import (
    STRUCTURAL_WATERSHED_STRATEGY,
    StructuralWatershedConfig,
    clamped_structural_watershed_config,
    run_structural_watershed,
    structural_watershed_mask,
)

SUBSTRATE = 40
FILL = 80
RIM = 220


def _component_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return count - 1


def _parallel_bright_traces() -> np.ndarray:
    image = np.full((160, 220), SUBSTRATE, np.uint8)
    image[20:140, 40:52] = FILL
    image[20:140, 40:43] = RIM
    image[20:140, 49:52] = RIM
    image[20:140, 70:82] = FILL
    image[20:140, 70:73] = RIM
    image[20:140, 79:82] = RIM
    image[20:140, 100:112] = FILL
    image[20:140, 100:103] = RIM
    image[20:140, 109:112] = RIM
    return cv2.GaussianBlur(image, (0, 0), 0.8)


def _wide_dark_plate() -> np.ndarray:
    image = np.full((180, 240), SUBSTRATE, np.uint8)
    image[30:150, 40:200] = 55
    image[30:150, 40:46] = RIM
    image[30:150, 194:200] = RIM
    image[30:36, 40:200] = RIM
    image[144:150, 40:200] = RIM
    return cv2.GaussianBlur(image, (0, 0), 0.8)


def _border_touching_trace() -> np.ndarray:
    image = np.full((120, 160), SUBSTRATE, np.uint8)
    image[:, 60:84] = FILL
    image[:, 60:64] = RIM
    image[:, 80:84] = RIM
    return cv2.GaussianBlur(image, (0, 0), 0.7)


def test_structural_strategy_token_is_recognised() -> None:
    assert normalize_metal_segmentation_strategy("structural_watershed") == STRUCTURAL_WATERSHED_STRATEGY
    assert normalize_metal_segmentation_strategy("Структурный водораздел") == STRUCTURAL_WATERSHED_STRATEGY


def test_parallel_traces_stay_separate_with_ridge_markers() -> None:
    image = _parallel_bright_traces()
    result = run_structural_watershed(
        image,
        GradientWatershedConfig(),
        clamped_structural_watershed_config(variant="s2"),
        check_presence=False,
    )

    assert result.mask[80, 46] > 0
    assert result.mask[80, 76] > 0
    assert result.mask[80, 106] > 0
    assert result.mask[80, 60] == 0
    assert result.mask[80, 90] == 0
    assert _component_count(result.mask) >= 3
    assert np.any(result.debug_images["metal_structural_ridge_markers"] > 0)


def test_wide_dark_interior_is_not_ridge_only() -> None:
    image = _wide_dark_plate()
    result = run_structural_watershed(
        image,
        GradientWatershedConfig(),
        clamped_structural_watershed_config(variant="s2"),
        check_presence=False,
    )

    assert result.mask[90, 120] > 0
    assert result.mask[90, 20] == 0
    assert np.any(result.debug_images["metal_structural_wide_interior_markers"] > 0)


def test_border_touching_conductor_is_kept() -> None:
    image = _border_touching_trace()
    mask = structural_watershed_mask(
        image,
        GradientWatershedConfig(),
        clamped_structural_watershed_config(variant="s2"),
        check_presence=False,
    )

    assert mask[5, 72] > 0
    assert mask[-5, 72] > 0
    assert mask[60, 20] == 0


def test_structural_pipeline_stays_at_native_resolution() -> None:
    image = np.full((700, 700), SUBSTRATE, np.uint8)
    image[80:620, 200:260] = FILL
    image[80:620, 200:206] = RIM
    image[80:620, 254:260] = RIM
    mask = seeded_segmentation_mask(
        image,
        "structural_watershed",
        GradientWatershedConfig(),
    )
    assert mask.shape == (700, 700)


def test_empty_frame_stays_empty() -> None:
    image = np.full((80, 80), 128, np.uint8)
    mask = structural_watershed_mask(image, GradientWatershedConfig(), StructuralWatershedConfig())
    assert not np.any(mask)


def test_s1_variant_uses_existing_markers_without_ridge_seeds() -> None:
    image = _parallel_bright_traces()
    result = run_structural_watershed(
        image,
        GradientWatershedConfig(),
        clamped_structural_watershed_config(variant="s1"),
        check_presence=False,
    )
    assert result.mask.shape == image.shape
    assert not np.any(result.debug_images["metal_structural_ridge_markers"])
    assert "metal_structural_boundary_cost" in result.debug_images


def test_s5_keeps_adjacent_instance_ids_without_background_gap() -> None:
    from contour.vision.metal_recovery.structural_watershed import (
        _finalize_instance_labels,
        _instance_watershed,
    )

    fg = np.zeros((80, 140), np.uint8)
    fg[20:60, 30:40] = 255
    fg[20:60, 90:100] = 255
    bg = np.zeros((80, 140), np.uint8)
    bg[:, 0:4] = 255
    bg[:, 136:140] = 255
    cost = np.zeros((80, 140), np.float32)
    _mask, markers = _instance_watershed(fg, bg, cost, grow=True)
    labels = _finalize_instance_labels(markers)
    left = int(labels[40, 35])
    right = int(labels[40, 95])
    mid = int(labels[40, 65])
    assert left > 0
    assert right > 0
    assert left != right
    assert mid in {0, left, right}


def test_remap_positive_ids_does_not_merge_touching_labels() -> None:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:, :4] = 7
    labels[:, 4:] = 9
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    lookup[ids] = np.arange(1, int(ids.size) + 1, dtype=np.int32)
    remapped = np.zeros_like(labels)
    positive = labels > 0
    remapped[positive] = lookup[labels[positive]]
    assert remapped[0, 0] != remapped[0, 7]
    assert {int(remapped[0, 0]), int(remapped[0, 7])} == {1, 2}

