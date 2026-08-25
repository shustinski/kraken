"""Built-in conductor recovery preset payloads."""

from __future__ import annotations

from typing import Any


def _metal_keys() -> tuple[str, ...]:
    return (
        "metal_preset",
        "metal_min_contrast",
        "metal_min_object_source_contrast",
        "metal_min_object_rim_contrast",
        "metal_min_object_rim_area_fraction",
        "metal_min_hole_source_contrast",
        "metal_min_hole_source_contrast_fraction",
        "metal_gap_bridge_px",
        "metal_speckle_removal_px",
        "metal_min_trace_width_px",
        "metal_max_trace_width_px",
        "metal_min_trace_length_px",
        "metal_min_area",
        "metal_min_perimeter",
        "metal_hierarchy_mode",
        "epsilon",
        "min_polygon_angle",
        "min_points",
        "metal_approximation_enabled",
        "metal_border_handling",
        "metal_segmentation_strategy",
        "metal_auto_contrast_step",
        "metal_auto_source_contrast_step",
        "metal_auto_directional_gap_bridge_px",
        "metal_auto_directional_gap_min_source_intensity",
        "metal_preprocess_subtract_background",
        "metal_preprocess_background_sigma_fraction",
        "metal_preprocess_clahe_clip",
        "metal_preprocess_clahe_grid",
        "metal_preprocess_denoise",
        "metal_watershed_smoothing_sigma",
        "metal_watershed_core_margin",
        "metal_watershed_groove_margin",
        "metal_watershed_rim_probe_px",
        "metal_watershed_seed_speckle_px",
        "metal_watershed_valley_span_px",
        "metal_watershed_valley_depth",
        "metal_adaptive_block_size",
        "metal_adaptive_c",
        "metal_adaptive_method",
        "metal_random_walker_beta",
        "metal_random_walker_iterations",
        "metal_graph_cut_iterations",
        "metal_reconstruction_erode_px",
        "metal_boundary_relief",
        "metal_boundary_background_sigma",
    )


def standard_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "standard",
        "metal_min_contrast": 50.0,
        "metal_min_object_source_contrast": 12.0,
        "metal_min_object_rim_contrast": 36.0,
        "metal_min_object_rim_area_fraction": 0.001,
        "metal_min_hole_source_contrast": 8.0,
        "metal_min_hole_source_contrast_fraction": 0.35,
        "metal_gap_bridge_px": 0,
        "metal_speckle_removal_px": 3,
        "metal_min_trace_width_px": 4.0,
        "metal_max_trace_width_px": None,
        "metal_min_area": 95.0,
        "metal_min_perimeter": 10.0,
        "metal_hierarchy_mode": "full",
        "epsilon": 1.0,
        "min_polygon_angle": 0.0,
        "min_points": 3,
        "metal_approximation_enabled": True,
        "metal_border_handling": "mark",
        "metal_segmentation_strategy": "auto",
        "metal_auto_contrast_step": 10.0,
        "metal_auto_source_contrast_step": 4.0,
        "metal_auto_directional_gap_bridge_px": 3,
        "metal_auto_directional_gap_min_source_intensity": 45.0,
        "metal_preprocess_subtract_background": True,
        "metal_preprocess_background_sigma_fraction": 0.05,
        "metal_preprocess_clahe_clip": 2.0,
        "metal_preprocess_clahe_grid": 8,
        "metal_preprocess_denoise": "low",
        "metal_watershed_seed_speckle_px": 4,
    }


def noisy_sem_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "noisy_sem",
        "metal_min_contrast": 50.0,
        "metal_min_object_source_contrast": 12.0,
        "metal_min_object_rim_contrast": 36.0,
        "metal_min_object_rim_area_fraction": 0.001,
        "metal_min_hole_source_contrast": 8.0,
        "metal_min_hole_source_contrast_fraction": 0.35,
        "metal_auto_source_contrast_step": 4.0,
        "metal_auto_directional_gap_bridge_px": 3,
        "metal_auto_directional_gap_min_source_intensity": 45.0,
        "metal_gap_bridge_px": 4,
        "metal_speckle_removal_px": 2,
        "metal_min_trace_width_px": 10.0,
        "metal_min_area": 85.0,
        "metal_min_perimeter": 40.0,
        "epsilon": 2.0,
    }


def thin_traces_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "thin_traces",
        "metal_min_contrast": 50.0,
        "metal_min_object_source_contrast": 12.0,
        "metal_min_object_rim_contrast": 36.0,
        "metal_min_object_rim_area_fraction": 0.001,
        "metal_min_hole_source_contrast": 8.0,
        "metal_min_hole_source_contrast_fraction": 0.35,
        "metal_auto_source_contrast_step": 4.0,
        "metal_auto_directional_gap_bridge_px": 3,
        "metal_auto_directional_gap_min_source_intensity": 45.0,
        "metal_gap_bridge_px": 2,
        "metal_speckle_removal_px": 1,
        "metal_min_trace_width_px": 4.0,
        "metal_max_trace_width_px": 24.0,
        "metal_min_area": 35.0,
        "metal_min_perimeter": 26.0,
        "epsilon": 1.8,
    }


def wide_fills_metal_preset_payload() -> dict[str, Any]:
    return {
        "metal_preset": "wide_fills",
        "metal_min_contrast": 50.0,
        "metal_min_object_source_contrast": 12.0,
        "metal_min_object_rim_contrast": 36.0,
        "metal_min_object_rim_area_fraction": 0.001,
        "metal_min_hole_source_contrast": 8.0,
        "metal_min_hole_source_contrast_fraction": 0.35,
        "metal_auto_source_contrast_step": 4.0,
        "metal_auto_directional_gap_bridge_px": 3,
        "metal_auto_directional_gap_min_source_intensity": 45.0,
        "metal_gap_bridge_px": 3,
        "metal_speckle_removal_px": 1,
        "metal_min_trace_width_px": 14.0,
        "metal_min_area": 100.0,
        "metal_min_perimeter": 42.0,
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
