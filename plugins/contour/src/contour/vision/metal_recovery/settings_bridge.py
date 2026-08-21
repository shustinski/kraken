from __future__ import annotations

from typing import Any

from .detector import MetalRecoveryConfig
from .gradient_watershed import clamped_gradient_watershed_config
from .segmentation import resolve_metal_segmentation_strategy


def _normalize_border_mode(value: Any) -> str:
    text = str(value or "mark").strip().lower()
    if text in {"ignore", "игнорировать", "skip"}:
        return "ignore"
    if text in {"accept", "принимать"}:
        return "accept"
    return "mark"


def _normalize_hierarchy_mode(value: Any) -> bool:
    """True = external-only contours (RETR_EXTERNAL)."""
    text = str(value or "full").strip().lower()
    return text in {"external", "outer", "внешние"}


def _non_negative_int(value: Any, *, default: int) -> int:
    if value is None:
        return max(0, int(default))
    return max(0, int(value))


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        return None
    return parsed


def metal_recovery_config_from_settings(settings: Any) -> MetalRecoveryConfig:
    gap_raw = getattr(settings, "metal_gap_bridge_px", None)
    if gap_raw is None:
        gap_raw = getattr(settings, "metal_morph_close_radius", 2)
    speckle_raw = getattr(settings, "metal_speckle_removal_px", None)
    if speckle_raw is None:
        speckle_raw = getattr(settings, "metal_morph_open_radius", 0)

    watershed = clamped_gradient_watershed_config(
        smoothing_sigma=float(getattr(settings, "metal_watershed_smoothing_sigma", 1.0) or 1.0),
        core_margin=float(getattr(settings, "metal_watershed_core_margin", 8.0) or 0.0),
        groove_margin=float(getattr(settings, "metal_watershed_groove_margin", 16.0) or 0.0),
        rim_probe_px=int(getattr(settings, "metal_watershed_rim_probe_px", 6) or 1),
        seed_speckle_px=int(getattr(settings, "metal_watershed_seed_speckle_px", 4) or 0),
        valley_span_px=int(getattr(settings, "metal_watershed_valley_span_px", 5) or 0),
        valley_depth=float(getattr(settings, "metal_watershed_valley_depth", 45.0) or 0.0),
        random_walker_beta=float(getattr(settings, "metal_random_walker_beta", 90.0) or 90.0),
        random_walker_iterations=int(getattr(settings, "metal_random_walker_iterations", 160) or 160),
        graph_cut_iterations=int(getattr(settings, "metal_graph_cut_iterations", 5) or 5),
        reconstruction_erode_px=int(getattr(settings, "metal_reconstruction_erode_px", 0) or 0),
        boundary_relief=float(getattr(settings, "metal_boundary_relief", 16.0) or 16.0),
        boundary_background_sigma=float(
            getattr(settings, "metal_boundary_background_sigma", 12.0) or 12.0
        ),
    )
    external_only = _normalize_hierarchy_mode(getattr(settings, "metal_hierarchy_mode", "full"))
    return MetalRecoveryConfig(
        min_contrast=max(
            1.0,
            min(
                255.0,
                float(
                    getattr(
                        settings,
                        "metal_min_contrast",
                        max(1.0, float(getattr(settings, "metal_contrast_bias", 50.0) or 50.0)),
                    )
                ),
            ),
        ),
        min_object_source_contrast=max(
            0.0,
            min(255.0, float(getattr(settings, "metal_min_object_source_contrast", 12.0))),
        ),
        min_object_rim_contrast=max(
            0.0,
            min(255.0, float(getattr(settings, "metal_min_object_rim_contrast", 36.0))),
        ),
        min_object_rim_area_fraction=max(
            0.000001,
            min(1.0, float(getattr(settings, "metal_min_object_rim_area_fraction", 0.001))),
        ),
        min_hole_source_contrast=max(
            0.0,
            min(255.0, float(getattr(settings, "metal_min_hole_source_contrast", 8.0))),
        ),
        min_hole_source_contrast_fraction=max(
            0.0,
            min(1.0, float(getattr(settings, "metal_min_hole_source_contrast_fraction", 0.35))),
        ),
        segmentation_strategy=resolve_metal_segmentation_strategy(
            getattr(settings, "metal_segmentation_strategy", "auto"),
            use_wide_conductor_gradient=bool(getattr(settings, "metal_use_wide_conductor_gradient", False)),
        ),
        auto_contrast_step=max(
            0.0,
            min(255.0, float(getattr(settings, "metal_auto_contrast_step", 10.0))),
        ),
        auto_source_contrast_step=max(
            0.0,
            min(255.0, float(getattr(settings, "metal_auto_source_contrast_step", 4.0))),
        ),
        auto_directional_gap_bridge_px=max(
            0,
            min(64, int(getattr(settings, "metal_auto_directional_gap_bridge_px", 3))),
        ),
        auto_directional_gap_min_source_intensity=max(
            0.0,
            min(
                255.0,
                float(
                    getattr(
                        settings,
                        "metal_auto_directional_gap_min_source_intensity",
                        45.0,
                    )
                ),
            ),
        ),
        gap_bridge_px=_non_negative_int(gap_raw, default=2),
        speckle_removal_px=_non_negative_int(speckle_raw, default=0),
        min_width_px=max(0.5, float(getattr(settings, "metal_min_trace_width_px", 8) or 8)),
        max_width_px=_optional_positive_float(getattr(settings, "metal_max_trace_width_px", None)),
        min_length_px=max(1.0, float(getattr(settings, "metal_min_trace_length_px", 8) or 8)),
        min_area=max(0.0, float(getattr(settings, "metal_min_area", 60) or 60)),
        max_area=_optional_positive_float(getattr(settings, "metal_max_area", None)),
        min_perimeter=max(0.0, float(getattr(settings, "metal_min_perimeter", 32) or 32)),
        max_perimeter=_optional_positive_float(getattr(settings, "metal_max_perimeter", None)),
        epsilon_simplify=max(0.0, float(getattr(settings, "epsilon", 2.0) or 2.0)),
        min_points=max(3, int(getattr(settings, "min_points", 4) or 4)),
        min_polygon_angle_deg=max(0.0, float(getattr(settings, "min_polygon_angle", 0.0) or 0.0)),
        approximation_enabled=bool(getattr(settings, "metal_approximation_enabled", True)),
        retrieval_external_only=external_only,
        retrieval_mode="RETR_EXTERNAL" if external_only else "RETR_TREE",
        approximation_mode=str(getattr(settings, "approximation_mode", "CHAIN_APPROX_SIMPLE")),
        border_mode=_normalize_border_mode(getattr(settings, "metal_border_handling", "mark")),
        min_inner_hole_area=max(0.0, float(getattr(settings, "min_inner_hole_area", 100.0) or 100.0)),
        min_component_area=max(
            0.0,
            float(getattr(settings, "metal_min_object_area", getattr(settings, "metal_min_area", 60)) or 60),
        ),
        preset_name=str(getattr(settings, "metal_preset", "standard") or "standard"),
        use_wide_conductor_gradient=bool(getattr(settings, "metal_use_wide_conductor_gradient", False)),
        watershed_smoothing_sigma=watershed.smoothing_sigma,
        watershed_core_margin=watershed.core_margin,
        watershed_groove_margin=watershed.groove_margin,
        watershed_rim_probe_px=watershed.rim_probe_px,
        watershed_seed_speckle_px=watershed.seed_speckle_px,
        watershed_valley_span_px=watershed.valley_span_px,
        watershed_valley_depth=watershed.valley_depth,
        random_walker_beta=watershed.random_walker_beta,
        random_walker_iterations=watershed.random_walker_iterations,
        graph_cut_iterations=watershed.graph_cut_iterations,
        reconstruction_erode_px=watershed.reconstruction_erode_px,
        boundary_relief=watershed.boundary_relief,
        boundary_background_sigma=watershed.boundary_background_sigma,
    )
