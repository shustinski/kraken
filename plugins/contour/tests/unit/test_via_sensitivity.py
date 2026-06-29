"""Tests for centralized via sensitivity profiles."""

from __future__ import annotations

from contour.application.processing import ContourExtractionSettings
from contour.application.via_sensitivity import via_sensitivity_profile, via_sensitivity_settings_patch


def test_medium_sensitivity_matches_tuned_defaults() -> None:
    profile = via_sensitivity_profile("medium")
    assert profile.via_white_range_min == 140
    assert profile.bright_via_hard_reject_on_asymmetry is True
    assert profile.bright_via_min_final_score == 42.0


def test_sensitivity_patch_keeps_asymmetry_gate_for_medium() -> None:
    patch = via_sensitivity_settings_patch("medium")
    settings = ContourExtractionSettings(**patch)
    assert settings.bright_via_hard_reject_on_asymmetry is True
    assert settings.via_white_range_min == 140
    assert settings.via_white_range_enabled is True


def test_low_sensitivity_is_stricter_than_high() -> None:
    low = via_sensitivity_profile("low")
    high = via_sensitivity_profile("high")
    assert low.via_white_range_min > high.via_white_range_min
    assert low.bright_via_min_final_score > high.bright_via_min_final_score
