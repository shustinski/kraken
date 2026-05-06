"""Built-in via-detection preset payloads.

Pure data — no widget state required.
"""

from __future__ import annotations


def noisy_traces_via_preset_payload() -> dict[str, object]:
    return {
        "via_search_mode": "template",
        "via_white_range_enabled": True,
        "via_white_range_min": 180,
        "via_white_range_max": 255,
        "via_black_range_enabled": False,
        "via_black_range_min": 0,
        "via_black_range_max": 30,
        "via_min_score": 0.50,
        "via_min_contrast": 30.0,
        "via_min_edge_coverage": 0.50,
        "via_spot_line_suppression": 0.85,
        "via_template_min_score": 0.35,
        "via_min_roundness": 50.0,
        "debug_enabled": True,
    }


def blurred_via_preset_payload() -> dict[str, object]:
    return {
        "via_search_mode": "template",
        "via_white_range_enabled": True,
        "via_white_range_min": 135,
        "via_white_range_max": 255,
        "via_black_range_enabled": False,
        "via_black_range_min": 0,
        "via_black_range_max": 45,
        "via_min_score": 0.30,
        "via_min_contrast": 10.0,
        "via_min_edge_coverage": 0.30,
        "via_spot_line_suppression": 0.55,
        "via_template_min_score": 0.30,
        "via_min_roundness": 25.0,
        "debug_enabled": True,
    }


def built_in_via_presets(language: str) -> dict[str, dict[str, object]]:
    """Return the built-in preset name → payload mapping for *language* (``"ru"`` or ``"en"``)."""
    if language == "ru":
        return {
            "Яркие via на дорожках": noisy_traces_via_preset_payload(),
            "Слабые/размытые via": blurred_via_preset_payload(),
        }
    return {
        "Bright vias on traces": noisy_traces_via_preset_payload(),
        "Weak/blurred vias": blurred_via_preset_payload(),
    }


__all__ = [
    "blurred_via_preset_payload",
    "built_in_via_presets",
    "noisy_traces_via_preset_payload",
]


def _standard_via_preset_payload() -> dict[str, object]:
    return {
        "via_search_mode": "heuristic",
        "bright_via_diameter_min": 6,
        "bright_via_diameter_max": 12,
        "via_search_sensitivity": "medium",
        "via_heuristic_polarity": "auto",
        "via_min_roundness": 40.0,
        "bright_via_min_final_score": 38.0,
    }


def built_in_via_presets(language: str) -> dict[str, dict[str, object]]:  # type: ignore[no-redef]
    standard = _standard_via_preset_payload()
    small = {**standard, "bright_via_diameter_min": 3, "bright_via_diameter_max": 7, "bright_via_nms_distance": 4}
    large = {**standard, "bright_via_diameter_min": 12, "bright_via_diameter_max": 28, "bright_via_nms_distance": 10}
    bright = {**standard, "via_search_mode": "bright_tophat_dog", "via_heuristic_polarity": "bright"}
    dark = {**standard, "via_heuristic_polarity": "dark"}
    ring = {**standard, "via_heuristic_polarity": "ring_light_ring", "via_min_roundness": 35.0}
    weak = {**blurred_via_preset_payload(), "via_search_mode": "heuristic", "via_search_sensitivity": "high", "bright_via_min_final_score": 28.0}
    if language == "ru":
        return {
            "Стандартный": standard,
            "Малые via": small,
            "Крупные via": large,
            "Светлые via": bright,
            "Тёмные via": dark,
            "Via с кольцом": ring,
            "Слабый контраст": weak,
        }
    return {
        "Standard": standard,
        "Small vias": small,
        "Large vias": large,
        "Bright vias": bright,
        "Dark vias": dark,
        "Ring vias": ring,
        "Weak contrast": weak,
    }
