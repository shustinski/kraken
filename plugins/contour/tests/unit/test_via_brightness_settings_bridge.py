"""Brightness range settings bridge for universal via detection."""

from __future__ import annotations

import cv2
import numpy as np

from contour.application.processing import ContourExtractionSettings
from contour.vision.integration import run_via_detection
from contour.vision.schemas import OutputShapeKind
from contour.vision.via.orchestrator import _detection_to_hit
from contour.vision.via_detection.config import ViaPolarity
from contour.vision.via_detection.heuristic_detector import detect_vias_heuristic
from contour.vision.via_detection.result import ViaDetection
from contour.vision.via_detection.settings_bridge import heuristic_config_from_settings


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


def test_fixed_size_uses_selected_diameter_instead_of_legacy_diameter_list() -> None:
    settings = ContourExtractionSettings(
        via_size_mode="fixed",
        bright_via_diameter_min=14,
        bright_via_diameter_max=14,
        via_fixed_diameters_text="6, 8, 10",
    )

    cfg = heuristic_config_from_settings(settings)

    assert cfg.allowed_diameters() == [14]


def test_fixed_size_forces_candidate_output_geometry() -> None:
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


def test_fixed_size_is_preserved_by_full_sem_via_detection() -> None:
    image = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(image, (24, 24), 5, 255, thickness=-1)
    settings = ContourExtractionSettings(
        algorithm_backend="sem",
        object_type="via",
        via_search_mode="heuristic",
        via_size_mode="fixed",
        bright_via_diameter_min=14,
        bright_via_diameter_max=14,
        via_fixed_diameters_text="6, 8, 10",
    )

    output = run_via_detection(
        image,
        image_path="frame.png",
        output_kind=OutputShapeKind.AXIS_ALIGNED_BOX,
        legacy_settings=settings,
    )

    assert len(output.hits) == 1
    assert output.hits[0].width == 14.0
    assert output.hits[0].height == 14.0


def test_local_contrast_recovers_subtle_via_within_absolute_brightness_range() -> None:
    image = np.full((96, 96), 145, dtype=np.uint8)
    cv2.circle(image, (48, 48), 5, 153, thickness=-1)
    settings = ContourExtractionSettings(
        via_size_mode="fixed",
        bright_via_diameter_min=10,
        bright_via_diameter_max=10,
        via_white_range_enabled=True,
        via_white_range_min=140,
    )

    result = detect_vias_heuristic(image, heuristic_config_from_settings(settings))

    assert len(result.accepted) == 1
    assert abs(result.accepted[0].x - 48.0) < 0.1
    assert abs(result.accepted[0].y - 48.0) < 0.1


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
