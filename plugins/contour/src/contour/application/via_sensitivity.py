"""Centralized via search sensitivity profiles (low / medium / high)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .processing import normalize_via_search_sensitivity


@dataclass(frozen=True, slots=True)
class ViaSensitivityProfileValues:
    bright_via_threshold_percentile: float
    via_white_range_min: int
    bright_via_min_final_score: float
    bright_via_min_circularity: float
    bright_via_min_isolation_score: float
    bright_via_hard_reject_on_asymmetry: bool
    bright_via_hard_reject_on_edge: bool
    bright_via_hard_reject_on_line: bool
    bright_via_max_radial_asymmetry: float = 32.0


_PROFILES: dict[str, ViaSensitivityProfileValues] = {
    "low": ViaSensitivityProfileValues(
        bright_via_threshold_percentile=99.5,
        via_white_range_min=160,
        bright_via_min_final_score=55.0,
        bright_via_min_circularity=0.40,
        bright_via_min_isolation_score=0.42,
        bright_via_hard_reject_on_asymmetry=True,
        bright_via_hard_reject_on_edge=True,
        bright_via_hard_reject_on_line=True,
        bright_via_max_radial_asymmetry=28.0,
    ),
    "medium": ViaSensitivityProfileValues(
        bright_via_threshold_percentile=99.0,
        via_white_range_min=140,
        bright_via_min_final_score=42.0,
        bright_via_min_circularity=0.30,
        bright_via_min_isolation_score=0.38,
        bright_via_hard_reject_on_asymmetry=True,
        bright_via_hard_reject_on_edge=False,
        bright_via_hard_reject_on_line=False,
        bright_via_max_radial_asymmetry=32.0,
    ),
    "high": ViaSensitivityProfileValues(
        bright_via_threshold_percentile=98.0,
        via_white_range_min=110,
        bright_via_min_final_score=32.0,
        bright_via_min_circularity=0.22,
        bright_via_min_isolation_score=0.35,
        bright_via_hard_reject_on_asymmetry=True,
        bright_via_hard_reject_on_edge=False,
        bright_via_hard_reject_on_line=False,
        bright_via_max_radial_asymmetry=40.0,
    ),
}


def via_sensitivity_profile(level: Any) -> ViaSensitivityProfileValues:
    key = normalize_via_search_sensitivity(level)
    return _PROFILES.get(key, _PROFILES["medium"])


def via_sensitivity_settings_patch(level: Any) -> dict[str, object]:
    """Settings fields updated when the user picks a sensitivity level."""
    key = normalize_via_search_sensitivity(level)
    profile = via_sensitivity_profile(key)
    return {
        "via_search_sensitivity": key,
        "bright_via_threshold_percentile": profile.bright_via_threshold_percentile,
        "via_white_range_enabled": True,
        "via_white_range_min": profile.via_white_range_min,
        "via_white_range_max": 255,
        "bright_via_bright_center_min_score": float(profile.via_white_range_min),
        "bright_via_min_final_score": profile.bright_via_min_final_score,
        "bright_via_min_circularity": profile.bright_via_min_circularity,
        "bright_via_min_isolation_score": profile.bright_via_min_isolation_score,
        "bright_via_hard_reject_on_asymmetry": profile.bright_via_hard_reject_on_asymmetry,
        "bright_via_hard_reject_on_edge": profile.bright_via_hard_reject_on_edge,
        "bright_via_hard_reject_on_line": profile.bright_via_hard_reject_on_line,
        "bright_via_max_radial_asymmetry": profile.bright_via_max_radial_asymmetry,
    }
