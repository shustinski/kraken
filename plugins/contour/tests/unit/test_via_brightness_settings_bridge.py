"""Brightness range settings bridge for universal via detection."""

from __future__ import annotations

import cv2
import numpy as np

from contour.application.processing import ContourExtractionSettings
from contour.vision.integration import run_via_detection
from contour.vision.schemas import OutputShapeKind
from contour.vision.via.orchestrator import _detection_to_hit
from contour.vision.via_detection.config import ViaPolarity
from contour.vision.via_detection.heuristic_detector import (
    _fast_percentile,
    _mean_mask,
    detect_vias_heuristic,
)
from contour.vision.via_detection.result import ViaDetection
from contour.vision.via_detection.settings_bridge import heuristic_config_from_settings


def test_hot_path_statistics_match_numpy() -> None:
    rng = np.random.default_rng(17)
    patch = rng.integers(0, 256, size=(31, 29), dtype=np.uint8)
    mask = np.zeros_like(patch, dtype=bool)
    mask[3:27, 5:23] = True
    gradients = rng.normal(size=417).astype(np.float32)

    assert np.isclose(_mean_mask(patch, mask), float(np.mean(patch[mask])))
    for percentile in (10.0, 50.0, 60.0, 88.0, 100.0):
        assert np.isclose(
            _fast_percentile(gradients, percentile),
            float(np.percentile(gradients, percentile)),
            rtol=1e-6,
            atol=1e-6,
        )


def test_heuristic_config_uses_white_range_for_bright_only() -> None:
    settings = ContourExtractionSettings(
        via_white_range_enabled=True,
        via_white_range_min=150,
        via_white_range_max=250,
        via_black_range_enabled=False,
    )
    cfg = heuristic_config_from_settings(settings)
    assert cfg.polarity == str(ViaPolarity.BRIGHT)
    assert cfg.bright_range_enabled is True
    assert cfg.bright_range_min == 150.0
    assert cfg.dark_range_enabled is False
    assert not hasattr(cfg, "max_seed_count")


def test_heuristic_config_uses_both_ranges_when_enabled() -> None:
    settings = ContourExtractionSettings(
        via_white_range_enabled=True,
        via_black_range_enabled=True,
        via_black_range_min=5,
        via_black_range_max=40,
    )
    cfg = heuristic_config_from_settings(settings)
    assert cfg.polarity == str(ViaPolarity.AUTO)
    assert cfg.dark_range_enabled is True
    assert cfg.dark_range_min == 5.0
    assert cfg.dark_range_max == 40.0


def test_candidate_search_always_uses_minimum_and_maximum_diameter() -> None:
    settings = ContourExtractionSettings(
        via_size_mode="fixed",
        bright_via_diameter_min=8,
        bright_via_diameter_max=14,
        via_output_diameter=21,
        via_fixed_diameters_text="6, 8, 10",
    )

    cfg = heuristic_config_from_settings(settings)

    assert cfg.diameter_mode == "range"
    assert cfg.allowed_diameters() == list(range(8, 15))


def test_output_size_forces_uniform_geometry() -> None:
    detection = ViaDetection(
        x=20.0,
        y=30.0,
        bbox=(15, 25, 10, 12),
        score=90.0,
        diameter_estimate=11.0,
        contrast=20.0,
        prominence=15.0,
        compactness=0.9,
        aspect=1.0,
    )

    hit = _detection_to_hit(detection, "heuristic", [14])

    assert hit.width == 14.0
    assert hit.height == 14.0


def test_output_size_is_independent_from_search_range_in_full_sem_detection() -> None:
    image = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(image, (24, 24), 5, 255, thickness=-1)
    settings = ContourExtractionSettings(
        algorithm_backend="sem",
        object_type="via",
        via_search_mode="heuristic",
        via_size_mode="range",
        bright_via_diameter_min=8,
        bright_via_diameter_max=14,
        via_output_diameter=17,
        via_fixed_diameters_text="6, 8, 10",
    )

    output = run_via_detection(
        image,
        image_path="frame.png",
        output_kind=OutputShapeKind.AXIS_ALIGNED_BOX,
        legacy_settings=settings,
    )

    assert len(output.hits) == 1
    assert output.hits[0].width == 17.0
    assert output.hits[0].height == 17.0


def test_local_contrast_recovers_subtle_via_within_absolute_brightness_range() -> None:
    image = np.full((96, 96), 145, dtype=np.uint8)
    cv2.circle(image, (48, 48), 5, 153, thickness=-1)
    settings = ContourExtractionSettings(
        via_size_mode="fixed",
        bright_via_diameter_min=10,
        bright_via_diameter_max=10,
        via_white_range_enabled=True,
        via_white_range_min=140,
        heuristic_min_center_contrast=1.0,
        heuristic_min_peak_prominence=1.0,
    )

    result = detect_vias_heuristic(image, heuristic_config_from_settings(settings))

    assert len(result.accepted) == 1
    assert abs(result.accepted[0].x - 48.0) < 0.1
    assert abs(result.accepted[0].y - 48.0) < 0.1


def test_white_range_generates_candidates_for_every_allowed_brightness() -> None:
    image = np.zeros((180, 180), dtype=np.uint8)
    expected: list[tuple[int, int]] = []
    for row, y in enumerate(range(10, 171, 10)):
        for column, x in enumerate(range(10, 171, 10)):
            cv2.circle(image, (x, y), 4, 150 if (row + column) % 2 else 245, thickness=-1)
            expected.append((x, y))
    settings = ContourExtractionSettings(
        via_size_mode="fixed",
        bright_via_diameter_min=7,
        bright_via_diameter_max=7,
        via_white_range_enabled=True,
        via_white_range_min=140,
        bright_via_min_final_score=0.0,
        heuristic_min_center_contrast=1.0,
        heuristic_min_peak_prominence=1.0,
        heuristic_min_compactness=0.01,
    )

    result = detect_vias_heuristic(image, heuristic_config_from_settings(settings))
    detected = np.asarray([(item.x, item.y) for item in result.accepted], dtype=np.float32)

    assert len(result.accepted) == len(expected)
    for x, y in expected:
        assert float(np.min(np.hypot(detected[:, 0] - x, detected[:, 1] - y))) <= 1.0


def test_white_brightness_range_remains_a_hard_gate() -> None:
    image = np.full((96, 96), 100, dtype=np.uint8)
    cv2.circle(image, (48, 48), 5, 112, thickness=-1)
    settings = ContourExtractionSettings(
        via_size_mode="fixed",
        bright_via_diameter_min=10,
        bright_via_diameter_max=10,
        via_white_range_enabled=True,
        via_white_range_min=140,
    )

    result = detect_vias_heuristic(image, heuristic_config_from_settings(settings))

    assert result.accepted == []
    assert any(item.reject_reason == "hard:brightness_range" for item in result.rejected)


def test_line_and_diffuse_bright_spot_are_not_accepted_as_vias() -> None:
    settings = ContourExtractionSettings(
        via_size_mode="fixed",
        bright_via_diameter_min=10,
        bright_via_diameter_max=10,
        via_white_range_enabled=True,
        via_white_range_min=140,
        # This test exercises rejection of diffuse spots specifically.
        # The product default is intentionally more permissive (0.2).
        heuristic_min_edge_sharpness=0.4,
    )
    config = heuristic_config_from_settings(settings)

    line = np.full((96, 96), 50, dtype=np.uint8)
    cv2.line(line, (5, 48), (90, 48), 220, thickness=3)
    assert detect_vias_heuristic(line, config).accepted == []

    yy, xx = np.indices((96, 96))
    diffuse = 50.0 + 180.0 * np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / (2.0 * 5.0**2))
    diffuse_result = detect_vias_heuristic(np.clip(diffuse, 0, 255).astype(np.uint8), config)
    assert diffuse_result.accepted == []
    assert any("diffuse_spot" in str(item.reject_reason) for item in diffuse_result.rejected)
