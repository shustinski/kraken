from __future__ import annotations

import cv2
import numpy as np
import pytest

from contour.application.processing import ContourExtractionSettings
from contour.vision.integration import run_via_detection
from contour.vision.schemas import OutputShapeKind
from contour.vision.via.bright_tophat_dog import (
    BrightViaDetection,
    BrightViaDetectorConfig,
    bright_center_score,
    detect_bright_vias,
    edge_likeness_score,
    line_likeness_score,
    mask_fraction,
    radial_symmetry_score,
    suppress_close_points,
)


def _synthetic_bright_vias() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = np.full((96, 120), 115, dtype=np.uint8)
    cv2.rectangle(image, (12, 16), (104, 80), 150, thickness=-1)
    for center in ((34, 34), (72, 56)):
        cv2.circle(image, center, 3, 235, thickness=-1, lineType=cv2.LINE_AA)
    noise = rng.normal(0, 5, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _synthetic_image_detector_config(**overrides: object) -> BrightViaDetectorConfig:
    """Defaults tuned so two 4px dots + Gaussian noise stay accepted and match expected centers."""
    base: dict[str, object] = {
        "use_metal_mask": False,
        "threshold_percentile": 98.5,
        "min_final_score": 32.0,
        "bright_center_min_score": 3.0,
    }
    base.update(overrides)
    return BrightViaDetectorConfig(**base).validated()


def test_detect_bright_vias_finds_synthetic_bright_spots() -> None:
    image = _synthetic_bright_vias()
    config = _synthetic_image_detector_config()

    result = detect_bright_vias(image, config)

    centers = [(d.center[0], d.center[1]) for d in result.detections]
    assert len(result.detections) >= 2
    tol = 4.0
    assert any(
        abs(x - 34) <= tol and abs(y - 34) <= tol for x, y in centers
    ), f"expected a detection near (34,34), got {centers}"
    assert any(
        abs(x - 72) <= tol and abs(y - 56) <= tol for x, y in centers
    ), f"expected a detection near (72,56), got {centers}"
    assert {
        "processed",
        "tophat",
        "dog",
        "via_mask",
        "candidate_mask",
        "raw_gray",
        "metal_mask",
        "radial_symmetry",
        "edge_likeness",
        "line_likeness",
        "distance_to_edge",
        "final_overlay",
    } <= set(result.debug_images)


def test_detect_bright_vias_is_reproducible() -> None:
    image = _synthetic_bright_vias()
    config = _synthetic_image_detector_config()

    first = detect_bright_vias(image, config)
    second = detect_bright_vias(image, config)

    assert [d.bbox for d in first.detections] == [d.bbox for d in second.detections]
    assert [round(d.final_score, 6) for d in first.detections] == [round(d.final_score, 6) for d in second.detections]


def test_bright_center_score_and_mask_fraction_helpers() -> None:
    image = np.full((32, 32), 100, dtype=np.uint8)
    cv2.circle(image, (16, 16), 3, 150, thickness=-1)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    assert bright_center_score(image, (16, 16), 6) > 20.0
    assert mask_fraction(mask, (2, 2, 4, 4)) == 1.0
    assert mask_fraction(mask, (0, 0, 10, 10)) == pytest.approx(0.16)


def test_structural_scores_separate_via_from_edge_and_line() -> None:
    via = np.full((64, 64), 95, dtype=np.uint8)
    cv2.circle(via, (32, 32), 4, 220, thickness=-1, lineType=cv2.LINE_AA)

    edge = np.full((64, 64), 90, dtype=np.uint8)
    edge[:, 32:] = 210

    line = np.full((64, 64), 95, dtype=np.uint8)
    cv2.line(line, (10, 32), (54, 32), 220, thickness=5, lineType=cv2.LINE_AA)

    assert radial_symmetry_score(via, 32, 32, 5) < radial_symmetry_score(edge, 32, 32, 5)
    assert edge_likeness_score(edge, 32, 32, 5) > edge_likeness_score(via, 32, 32, 5)
    assert line_likeness_score(line, 32, 32, 5) > line_likeness_score(via, 32, 32, 5)


def test_detect_bright_vias_rejects_bright_line_segment() -> None:
    image = np.full((80, 100), 110, dtype=np.uint8)
    cv2.rectangle(image, (10, 24), (90, 56), 145, thickness=-1)
    cv2.line(image, (26, 40), (74, 40), 235, thickness=5, lineType=cv2.LINE_AA)

    result = detect_bright_vias(
        image,
        BrightViaDetectorConfig(
            use_metal_mask=False,
            threshold_percentile=98.0,
            max_line_likeness=60.0,
            max_edge_likeness=45.0,
            min_final_score=75.0,
        ),
    )

    assert result.detections == []


def test_disabled_metal_constraint_keeps_detection_independent_from_metal_mask() -> None:
    image = _synthetic_bright_vias()

    disabled = detect_bright_vias(
        image,
        _synthetic_image_detector_config(metal_constraint_mode="disabled"),
    )

    assert len(disabled.detections) >= 2
    assert all(detection.metal_fraction == 1.0 for detection in disabled.detections)


def test_suppress_close_points_keeps_highest_score() -> None:
    low = BrightViaDetection(
        center=(10.0, 10.0),
        bbox=(7, 7, 6, 6),
        area=20.0,
        circularity=0.8,
        aspect=1.0,
        brightness_score=6.0,
        local_peak_score=10.0,
        tophat_response=80.0,
        dog_response=70.0,
        metal_fraction=1.0,
        final_score=50.0,
        status="accepted",
    )
    high = BrightViaDetection(
        center=(12.0, 10.0),
        bbox=(9, 7, 6, 6),
        area=20.0,
        circularity=0.8,
        aspect=1.0,
        brightness_score=6.0,
        local_peak_score=10.0,
        tophat_response=90.0,
        dog_response=80.0,
        metal_fraction=1.0,
        final_score=70.0,
        status="accepted",
    )
    far = BrightViaDetection(
        center=(40.0, 40.0),
        bbox=(37, 37, 6, 6),
        area=20.0,
        circularity=0.8,
        aspect=1.0,
        brightness_score=6.0,
        local_peak_score=10.0,
        tophat_response=90.0,
        dog_response=80.0,
        metal_fraction=1.0,
        final_score=60.0,
        status="accepted",
    )

    kept = suppress_close_points([low, high, far], distance=5)

    assert kept == [high, far]


def test_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        BrightViaDetectorConfig(diameter_min=9, diameter_max=8).validated()
    with pytest.raises(ValueError):
        BrightViaDetectorConfig(median_blur_kernel=2).validated()
    with pytest.raises(ValueError):
        BrightViaDetectorConfig(dog_sigma_small=2.0, dog_sigma_large=1.0).validated()
    with pytest.raises(ValueError):
        BrightViaDetectorConfig(threshold_percentile=89.0).validated()
    with pytest.raises(ValueError):
        BrightViaDetectorConfig(max_radial_asymmetry=-1.0).validated()


def test_heuristic_via_mode_integrates_with_via_output() -> None:
    image = _synthetic_bright_vias()
    settings = ContourExtractionSettings(
        extraction_profile="vias",
        object_type="via",
        output_mode="box",
        via_search_mode="heuristic",
        bright_via_diameter_min=6,
        bright_via_diameter_max=16,
        bright_via_min_final_score=10.0,
        heuristic_min_center_contrast=1.0,
        heuristic_min_peak_prominence=1.0,
        heuristic_min_compactness=0.01,
    )

    output = run_via_detection(
        image,
        image_path="synthetic.png",
        output_kind=OutputShapeKind.AXIS_ALIGNED_BOX,
        legacy_settings=settings,
    )

    assert output.selected_strategy == "heuristic"
    assert len(output.hits) >= 1
