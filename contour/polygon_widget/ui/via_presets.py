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
