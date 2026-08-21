"""Golden tests: metal recovery on SEM frames vs reference CIF polygons."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from contour.application.processing import ContourExtractionSettings
from contour.domain.polygon_ring import is_valid_closed_polygon_ring
from contour.serializers import load_polygons_cif
from contour.ui.metal_presets import standard_metal_preset_payload
from contour.vision.metal_recovery import detect_metalization, metal_recovery_config_from_settings
from contour.vision.preprocessing import metal_preprocess_config_from_settings, preprocess_for_sem

_TEST_METAL_ROOT = Path(__file__).resolve().parent.parent / "test_metal"

_OCTA1 = "OCTA1_PECVD_M2_BS_0785"
_OCTA1_POLYGON_COUNT = 744
_OCTA1_MIN_MASK_IOU = 0.80
_OCTA1_MIN_POLYGON_IOU = 0.78


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


def _detected_conductor_mask(polygons, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if polygon.is_hole or polygon.shape_hint == "box":
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
    )
    return ContourExtractionSettings.from_dict(
        {**base.to_dict(), **standard_metal_preset_payload()},
    )


def _detect_conductors(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        pytest.skip(f"SEM frame missing: {image_path}")
    settings = _conductor_settings()
    preprocessed = preprocess_for_sem(image, metal_preprocess_config_from_settings(settings))
    result = detect_metalization(preprocessed, metal_recovery_config_from_settings(settings))
    return image, result


def test_octa1_golden_topology_is_repaired_not_rejected() -> None:
    image_path = _TEST_METAL_ROOT / "jpg" / f"{_OCTA1}.jpg"
    _, result = _detect_conductors(image_path)

    assert result.accepted, "expected accepted conductors"
    invalid = [
        polygon.id
        for polygon in result.accepted
        if not is_valid_closed_polygon_ring(polygon.points)
        and len(polygon.points) <= 192
    ]
    assert len(invalid) / max(1, len(result.accepted)) <= 0.05, f"invalid topology ids: {invalid[:8]}"


def test_octa1_golden_mask_iou() -> None:
    image_path = _TEST_METAL_ROOT / "jpg" / f"{_OCTA1}.jpg"
    cif_path = _TEST_METAL_ROOT / "cif" / f"{_OCTA1}.cif"
    if not cif_path.is_file():
        pytest.skip(f"Reference CIF missing: {cif_path}")

    image, result = _detect_conductors(image_path)
    _image_name, _image_size, reference_polygons = load_polygons_cif(cif_path)
    reference_mask = _reference_conductor_mask(reference_polygons, image.shape[:2])
    detected_mask = result.debug_images["metal_binary_mask"]

    iou = _mask_iou(reference_mask, detected_mask)
    assert iou >= _OCTA1_MIN_MASK_IOU, f"mask IoU {iou:.3f} < {_OCTA1_MIN_MASK_IOU:.3f}"


def test_octa1_golden_polygon_union_iou() -> None:
    image_path = _TEST_METAL_ROOT / "jpg" / f"{_OCTA1}.jpg"
    cif_path = _TEST_METAL_ROOT / "cif" / f"{_OCTA1}.cif"
    if not cif_path.is_file():
        pytest.skip(f"Reference CIF missing: {cif_path}")

    image, result = _detect_conductors(image_path)
    _image_name, _image_size, reference_polygons = load_polygons_cif(cif_path)
    reference_mask = _reference_conductor_mask(reference_polygons, image.shape[:2])
    detected_mask = _detected_conductor_mask(result.accepted, image.shape[:2])

    iou = _mask_iou(reference_mask, detected_mask)
    assert iou >= _OCTA1_MIN_POLYGON_IOU, (
        f"polygon union IoU {iou:.3f} < {_OCTA1_MIN_POLYGON_IOU:.3f}"
    )


def test_octa1_golden_polygon_count_is_reasonable() -> None:
    image_path = _TEST_METAL_ROOT / "jpg" / f"{_OCTA1}.jpg"
    cif_path = _TEST_METAL_ROOT / "cif" / f"{_OCTA1}.cif"
    if not cif_path.is_file():
        pytest.skip(f"Reference CIF missing: {cif_path}")

    _, result = _detect_conductors(image_path)
    _image_name, _image_size, reference_polygons = load_polygons_cif(cif_path)
    reference_count = sum(1 for polygon in reference_polygons if polygon.shape_hint != "box")

    detected_count = sum(
        1 for polygon in result.accepted if not polygon.is_hole and polygon.shape_hint != "box"
    )
    assert detected_count > 0
    assert abs(detected_count - reference_count) / max(reference_count, 1) <= 0.35
