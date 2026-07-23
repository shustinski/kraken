"""Regression tests for bright via detection on real SEM fixtures."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from contour.application.processing import ContourExtractionSettings
from contour.vision.integration import run_via_detection
from contour.vision.schemas import OutputShapeKind
from contour.vision.via.bright_tophat_dog import BrightViaDetectorConfig, detect_bright_vias
from contour.vision.via_detection import HeuristicViaDetectorConfig, detect_vias_heuristic

_TEST_VIA_ROOT = Path(__file__).resolve().parent.parent / "test_via"
_FIXTURES = ("KEEL_3W4_BS_12749", "KALIBR3_2W3_10557")
_MATCH_TOLERANCE_PX = 6.0


def _cif_ground_truth_centers(cif_path: Path) -> list[tuple[float, float]]:
    """Read expected result coordinates; the detector never receives this data."""

    if not cif_path.is_file():
        pytest.skip(f"CIF fixture not found: {cif_path}")
    payload = cif_path.read_text(encoding="utf-8", errors="replace")
    size_match = re.search(r"\(\s*S\s+(\d+)\s+(\d+)\s*\)", payload)
    if size_match is None:
        raise AssertionError(f"image size is missing in {cif_path}")
    image_height = int(size_match.group(2))
    boxes = re.findall(r"^B\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s*;", payload, flags=re.MULTILINE)
    return [(float(x), float(image_height - int(y))) for _width, _height, x, y in boxes]


def _detection_metrics(
    ground_truth: list[tuple[float, float]],
    detected: list[tuple[float, float]],
    *,
    tolerance: float = _MATCH_TOLERANCE_PX,
) -> tuple[float, float, int, int]:
    if not ground_truth:
        return 1.0, 1.0, 0, len(detected)
    pairs = sorted(
        (float(np.hypot(dx - gx, dy - gy)), gt_index, det_index)
        for gt_index, (gx, gy) in enumerate(ground_truth)
        for det_index, (dx, dy) in enumerate(detected)
        if float(np.hypot(dx - gx, dy - gy)) <= tolerance
    )
    matched_gt: set[int] = set()
    matched_det: set[int] = set()
    for _distance, gt_index, det_index in pairs:
        if gt_index in matched_gt or det_index in matched_det:
            continue
        matched_gt.add(gt_index)
        matched_det.add(det_index)
    true_positives = len(matched_gt)
    false_positives = len(detected) - len(matched_det)
    recall = true_positives / len(ground_truth)
    precision = true_positives / len(detected) if detected else 0.0
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


def _heuristic_via_preset_config() -> HeuristicViaDetectorConfig:
    return HeuristicViaDetectorConfig(
        diameter_min=6,
        diameter_max=12,
        polarity="bright",
        seed_percentile=90.0,
        nms_distance=5,
        bright_range_enabled=True,
        bright_range_min=100.0,
    )


@pytest.mark.parametrize("stem", _FIXTURES)
def test_bright_via_regression_fixture_exists(stem: str) -> None:
    img_path = _TEST_VIA_ROOT / "img" / f"{stem}.jpg"
    mask_path = _TEST_VIA_ROOT / "mask" / f"{stem}.jpg"
    if not img_path.is_file():
        pytest.skip(f"image not found: {img_path}")
    if not mask_path.is_file():
        pytest.skip(f"mask not found: {mask_path}")
    gt = _cif_ground_truth_centers(_TEST_VIA_ROOT / "cif" / f"{stem}.cif")
    assert len(gt) > 0


@pytest.mark.parametrize("stem", _FIXTURES)
def test_binary_mask_recovers_every_cif_via(stem: str) -> None:
    mask = cv2.imread(str(_TEST_VIA_ROOT / "mask" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    ground_truth = _cif_ground_truth_centers(_TEST_VIA_ROOT / "cif" / f"{stem}.cif")
    result = detect_bright_vias(mask, _bright_via_preset_config())
    detected = [item.center for item in result.detections]
    recall, precision, false_positives, _count = _detection_metrics(ground_truth, detected)
    assert recall == 1.0
    assert precision == 1.0
    assert false_positives == 0


@pytest.mark.parametrize("stem", _FIXTURES)
def test_heuristic_binary_mask_recovers_every_cif_via(stem: str) -> None:
    mask = cv2.imread(str(_TEST_VIA_ROOT / "mask" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    ground_truth = _cif_ground_truth_centers(_TEST_VIA_ROOT / "cif" / f"{stem}.cif")
    result = detect_vias_heuristic(mask, _heuristic_via_preset_config())
    detected = [(item.x, item.y) for item in result.accepted]
    recall, precision, false_positives, _count = _detection_metrics(ground_truth, detected)
    assert recall == 1.0
    assert precision == 1.0
    assert false_positives == 0


@pytest.mark.parametrize(
    ("stem", "min_recall", "min_precision", "max_false_positives"),
    [
        ("KEEL_3W4_BS_12749", 0.98, 0.91, 92),
        # Medium sensitivity is intentionally uncapped: all qualifying seeds are
        # evaluated, including the weaker KALIBR candidates formerly hidden by
        # the 1,400-seed ceiling.
        ("KALIBR3_2W3_10557", 1.0, 0.52, 130),
    ],
)
def test_heuristic_real_frame_quality(
    stem: str,
    min_recall: float,
    min_precision: float,
    max_false_positives: int,
) -> None:
    image = cv2.imread(str(_TEST_VIA_ROOT / "img" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    ground_truth = _cif_ground_truth_centers(_TEST_VIA_ROOT / "cif" / f"{stem}.cif")
    result = run_via_detection(
        image,
        image_path=None,
        output_kind=OutputShapeKind.AXIS_ALIGNED_BOX,
        legacy_settings=ContourExtractionSettings(
            algorithm_backend="sem",
            extraction_profile="vias",
            object_type="via",
            output_mode="box",
            via_search_mode="heuristic",
            via_size_mode="range",
            bright_via_diameter_min=6,
            bright_via_diameter_max=12,
            via_white_range_enabled=True,
            via_white_range_min=100,
        ),
    )
    detected = [(item.center_x, item.center_y) for item in result.hits]
    recall, precision, false_positives, _count = _detection_metrics(ground_truth, detected, tolerance=7.0)
    assert recall >= min_recall
    assert precision >= min_precision
    assert false_positives <= max_false_positives
    nearest_offsets = [
        min(
            ((dx - gx, dy - gy) for dx, dy in detected),
            key=lambda offset: float(np.hypot(offset[0], offset[1])),
        )
        for gx, gy in ground_truth
    ]
    matched_offsets = np.asarray(
        [offset for offset in nearest_offsets if float(np.hypot(offset[0], offset[1])) <= 7.0],
        dtype=np.float64,
    )
    assert abs(float(np.median(matched_offsets[:, 0]))) < 0.35
    assert abs(float(np.median(matched_offsets[:, 1]))) < 0.35
    assert float(np.median(np.hypot(matched_offsets[:, 0], matched_offsets[:, 1]))) < 0.8


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
    ground_truth = _cif_ground_truth_centers(_TEST_VIA_ROOT / "cif" / f"{stem}.cif")
    result = detect_bright_vias(image, _bright_via_preset_config())
    detected = [(d.center[0], d.center[1]) for d in result.detections]
    recall, precision, fp, _ = _detection_metrics(ground_truth, detected)

    assert recall >= min_recall, f"{stem}: recall {recall:.3f} < {min_recall}"
    assert precision >= min_precision, f"{stem}: precision {precision:.3f} < {min_precision}"
    assert fp <= max_fp, f"{stem}: false positives {fp} > {max_fp}"
