"""Brightness range settings bridge for universal via detection."""

from __future__ import annotations

from contour.application.processing import ContourExtractionSettings
from contour.vision.via_detection.config import ViaPolarity
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
