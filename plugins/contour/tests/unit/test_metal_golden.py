"""Golden tests: metal recovery on SEM frames vs reference CIF polygons."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from contour.application.processing import ContourExtractionSettings
from contour.serializers import load_polygons_cif
from contour.ui.metal_presets import standard_metal_preset_payload
from contour.vision.metal_recovery import detect_metalization, metal_recovery_config_from_settings
from contour.vision.metal_recovery.detector import _valid_topology

_TEST_METAL_ROOT = Path(__file__).resolve().parent.parent / "test_metal"

_GOLDEN_CASES = (
    pytest.param(
        "OCTA1_PECVD_M2_BS_0785",
        744,
        0.78,
        id="octa1",
    ),
    pytest.param(
        "MSP430_M1_BS_03720",
        710,
        0.70,
        id="msp430",
    ),
)


def _mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    first_active = first_mask > 0
    second_active = second_mask > 0
    union = int(np.logical_or(first_active, second_active).sum())
    if union <= 0:
        return 0.0
    intersection = int(np.logical_and(first_active, second_active).sum())
    return float(intersection / union)


def _reference_conductor_mask(polygons, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if polygon.shape_hint == "box":
            continue
        points = np.array(polygon.points, dtype=np.int32).reshape(-1, 1, 2)
        if points.shape[0] < 3:
            continue
        cv2.fillPoly(mask, [points], 255)
    return mask


def _conductor_settings() -> ContourExtractionSettings:
    base = ContourExtractionSettings(
        object_type="conductor",
        extraction_profile="conductors",
        recognition_mode="conductors",
        metal_structural_pipeline=True,
        algorithm_backend="legacy",
        metal_check_contour_validity=True,
    )
    return ContourExtractionSettings.from_dict(
        {**base.to_dict(), **standard_metal_preset_payload()},
    )


def _detect_conductors(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        pytest.skip(f"SEM frame missing: {image_path}")
    settings = _conductor_settings()
    result = detect_metalization(image, metal_recovery_config_from_settings(settings))
    return image, result


@pytest.mark.parametrize("stem,reference_count,min_mask_iou", _GOLDEN_CASES)
def test_metal_golden_topology_is_simple(stem: str, reference_count: int, min_mask_iou: float) -> None:
    image_path = _TEST_METAL_ROOT / "jpg" / f"{stem}.jpg"
    _, result = _detect_conductors(image_path)

    for polygon in result.accepted:
        valid, reason = _valid_topology(polygon.points, enabled=True)
        assert valid, f"{stem}: polygon id={polygon.id} has invalid topology: {reason}"


@pytest.mark.parametrize("stem,reference_count,min_mask_iou", _GOLDEN_CASES)
def test_metal_golden_mask_matches_reference(stem: str, reference_count: int, min_mask_iou: float) -> None:
    image_path = _TEST_METAL_ROOT / "jpg" / f"{stem}.jpg"
    cif_path = _TEST_METAL_ROOT / "cif" / f"{stem}.cif"
    if not cif_path.is_file():
        pytest.skip(f"Reference CIF missing: {cif_path}")

    image, result = _detect_conductors(image_path)
    _image_name, _image_size, reference_polygons = load_polygons_cif(cif_path)
    reference_mask = _reference_conductor_mask(reference_polygons, image.shape[:2])
    detected_mask = result.debug_images["metal_binary_mask"]

    iou = _mask_iou(reference_mask, detected_mask)
    assert iou >= min_mask_iou, f"{stem}: mask IoU {iou:.3f} < {min_mask_iou:.3f}"


@pytest.mark.parametrize("stem,reference_count,min_mask_iou", _GOLDEN_CASES)
def test_metal_golden_polygon_count_near_reference(stem: str, reference_count: int, min_mask_iou: float) -> None:
    image_path = _TEST_METAL_ROOT / "jpg" / f"{stem}.jpg"
    _, result = _detect_conductors(image_path)

    solid = [
        polygon
        for polygon in result.accepted
        if not polygon.is_hole and polygon.category in {"conductor", "metal_border"}
    ]
    lower = max(int(reference_count * 0.88), reference_count - 80)
    upper = int(reference_count * 1.08) + 5
    assert lower <= len(solid) <= upper, (
        f"{stem}: expected {lower}..{upper} solid polygons, got {len(solid)} (reference {reference_count})"
    )
