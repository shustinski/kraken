"""Built-in conductor recovery preset payloads."""

from __future__ import annotations

from typing import Any


def _metal_keys() -> tuple[str, ...]:
    return (
        "metal_preset",
        "metal_noise_suppression",
        "metal_contrast_bias",
        "metal_segmentation_strategy",
        "metal_gap_bridge_px",
        "metal_speckle_removal_px",
        "metal_contour_smooth_px",
        "metal_min_trace_width_px",
        "metal_max_trace_width_px",
        "metal_min_trace_length_px",
        "metal_min_straightness",
        "metal_min_area",
        "metal_min_perimeter",
        "metal_use_wide_conductor_gradient",
        "metal_allowed_angles",
        "metal_angle_tolerance_deg",
        "epsilon",
    )


def standard_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "standard",
        "metal_noise_suppression": 20,
        "metal_contrast_bias": 0.0,
        "metal_segmentation_strategy": "auto",
        "metal_gap_bridge_px": 2,
        "metal_speckle_removal_px": 0,
        "metal_contour_smooth_px": 0.0,
        "metal_min_trace_width_px": 8.0,
        "metal_max_trace_width_px": None,
        "metal_min_trace_length_px": 8.0,
        "metal_min_straightness": 0.2,
        "metal_min_area": 60.0,
        "metal_min_perimeter": 32.0,
        "metal_use_wide_conductor_gradient": False,
        "metal_allowed_angles": "free",
        "metal_angle_tolerance_deg": 7.0,
        "epsilon": 2.0,
    }


def noisy_sem_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "noisy_sem",
        "metal_noise_suppression": 70,
        "metal_contrast_bias": -15.0,
        "metal_segmentation_strategy": "auto",
        "metal_gap_bridge_px": 4,
        "metal_speckle_removal_px": 2,
        "metal_contour_smooth_px": 1.5,
        "metal_min_trace_width_px": 10.0,
        "metal_max_trace_width_px": None,
        "metal_min_trace_length_px": 32.0,
        "metal_min_straightness": 0.65,
        "metal_min_area": 85.0,
        "metal_min_perimeter": 40.0,
        "metal_use_wide_conductor_gradient": False,
        "metal_allowed_angles": "free",
        "metal_angle_tolerance_deg": 10.0,
        "epsilon": 2.0,
    }


def thin_traces_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "thin_traces",
        "metal_noise_suppression": 40,
        "metal_contrast_bias": 10.0,
        "metal_segmentation_strategy": "local_adaptive",
        "metal_gap_bridge_px": 2,
        "metal_speckle_removal_px": 1,
        "metal_contour_smooth_px": 1.0,
        "metal_min_trace_width_px": 4.0,
        "metal_max_trace_width_px": 24.0,
        "metal_min_trace_length_px": 28.0,
        "metal_min_straightness": 0.55,
        "metal_min_area": 35.0,
        "metal_min_perimeter": 26.0,
        "metal_use_wide_conductor_gradient": False,
        "metal_allowed_angles": "free",
        "metal_angle_tolerance_deg": 6.0,
        "epsilon": 1.8,
    }


def wide_fills_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "wide_fills",
        "metal_noise_suppression": 30,
        "metal_contrast_bias": -5.0,
        "metal_segmentation_strategy": "sauvola",
        "metal_gap_bridge_px": 3,
        "metal_speckle_removal_px": 1,
        "metal_contour_smooth_px": 2.0,
        "metal_min_trace_width_px": 14.0,
        "metal_max_trace_width_px": None,
        "metal_min_trace_length_px": 24.0,
        "metal_min_straightness": 0.45,
        "metal_min_area": 100.0,
        "metal_min_perimeter": 42.0,
        "metal_use_wide_conductor_gradient": True,
        "metal_allowed_angles": "free",
        "metal_angle_tolerance_deg": 8.0,
        "epsilon": 2.5,
    }


def built_in_metal_presets(language: str) -> dict[str, dict[str, Any]]:
    if language == "ru":
        return {
            "Стандартный": standard_metal_preset_payload(),
            "Шумное SEM": noisy_sem_metal_preset_payload(),
            "Тонкие дорожки": thin_traces_metal_preset_payload(),
            "Широкие заливки": wide_fills_metal_preset_payload(),
        }
    return {
        "Standard": standard_metal_preset_payload(),
        "Noisy SEM": noisy_sem_metal_preset_payload(),
        "Thin traces": thin_traces_metal_preset_payload(),
        "Wide fills": wide_fills_metal_preset_payload(),
    }


def metal_preset_table() -> dict[str, dict[str, Any]]:
    """Preset key → payload (builtin scenario keys)."""
    return {
        "standard": standard_metal_preset_payload(),
        "noisy_sem": noisy_sem_metal_preset_payload(),
        "thin_traces": thin_traces_metal_preset_payload(),
        "wide_fills": wide_fills_metal_preset_payload(),
    }


__all__ = [
    "built_in_metal_presets",
    "metal_preset_table",
    "noisy_sem_metal_preset_payload",
    "standard_metal_preset_payload",
    "thin_traces_metal_preset_payload",
    "wide_fills_metal_preset_payload",
]
