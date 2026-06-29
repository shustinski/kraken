"""Regression tests for bright via detection on real SEM fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from contour.vision.via.bright_tophat_dog import BrightViaDetectorConfig, detect_bright_vias

_TEST_VIA_ROOT = Path(__file__).resolve().parent.parent / "test_via"
_FIXTURES = ("KEEL_3W4_BS_12749", "KALIBR3_2W3_10557")
_MATCH_TOLERANCE_PX = 6.0


def _mask_ground_truth_centers(mask_path: Path) -> list[tuple[float, float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        pytest.skip(f"mask not found: {mask_path}")
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: list[tuple[float, float]] = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            centers.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    return centers


def _detection_metrics(
    ground_truth: list[tuple[float, float]],
    detected: list[tuple[float, float]],
    *,
    tolerance: float = _MATCH_TOLERANCE_PX,
) -> tuple[float, float, int, int]:
    if not ground_truth:
        return 1.0, 1.0, 0, len(detected)
    matched_gt = sum(
        1
        for gx, gy in ground_truth
        if any(abs(dx - gx) <= tolerance and abs(dy - gy) <= tolerance for dx, dy in detected)
    )
    false_positives = sum(
        1
        for dx, dy in detected
        if not any(abs(dx - gx) <= tolerance and abs(dy - gy) <= tolerance for gx, gy in ground_truth)
    )
    recall = matched_gt / len(ground_truth)
    precision = matched_gt / len(detected) if detected else 0.0
    return recall, precision, false_positives, len(detected)


def _bright_via_preset_config() -> BrightViaDetectorConfig:
    """User-like settings: bright SEM vias, diameter 6–12, NMS 10."""
    return BrightViaDetectorConfig(
        diameter_min=6,
        diameter_max=12,
        min_final_score=42.0,
        nms_distance=10,
        use_metal_mask=False,
        threshold_percentile=99.0,
        bright_range_enabled=True,
        bright_range_min=140.0,
        bright_range_max=255.0,
        min_circularity=0.30,
        min_isolation_score=0.38,
        min_annular_contrast=6.0,
        hard_reject_on_asymmetry=True,
        max_radial_asymmetry=32.0,
    ).validated()


@pytest.mark.parametrize("stem", _FIXTURES)
def test_bright_via_regression_fixture_exists(stem: str) -> None:
    img_path = _TEST_VIA_ROOT / "img" / f"{stem}.jpg"
    mask_path = _TEST_VIA_ROOT / "mask" / f"{stem}.jpg"
    if not img_path.is_file():
        pytest.skip(f"image not found: {img_path}")
    if not mask_path.is_file():
        pytest.skip(f"mask not found: {mask_path}")
    gt = _mask_ground_truth_centers(mask_path)
    assert len(gt) > 0


@pytest.mark.parametrize(
    ("stem", "min_recall", "min_precision", "max_fp"),
    [
        ("KEEL_3W4_BS_12749", 0.89, 0.88, 110),
        ("KALIBR3_2W3_10557", 0.63, 0.58, 60),
    ],
)
def test_bright_via_regression_quality_targets(
    stem: str,
    min_recall: float,
    min_precision: float,
    max_fp: int,
) -> None:
    img_path = _TEST_VIA_ROOT / "img" / f"{stem}.jpg"
    mask_path = _TEST_VIA_ROOT / "mask" / f"{stem}.jpg"
    if not img_path.is_file() or not mask_path.is_file():
        pytest.skip("fixture images not available")

    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    ground_truth = _mask_ground_truth_centers(mask_path)
    result = detect_bright_vias(image, _bright_via_preset_config())
    detected = [(d.center[0], d.center[1]) for d in result.detections]
    recall, precision, fp, _ = _detection_metrics(ground_truth, detected)

    assert recall >= min_recall, f"{stem}: recall {recall:.3f} < {min_recall}"
    assert precision >= min_precision, f"{stem}: precision {precision:.3f} < {min_precision}"
    assert fp <= max_fp, f"{stem}: false positives {fp} > {max_fp}"
