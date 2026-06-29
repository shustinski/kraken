from __future__ import annotations

from typing import Any

from .detector import MetalRecoveryConfig
from .segmentation import normalize_metal_segmentation_strategy

_ALLOWED_MAP = {
    "90_only": "90_only",
    "90": "90_only",
    "ortho": "90_only",
    "45_90": "45_90",
    "45": "45_90",
    "free": "free",
    "arbitrary": "free",
    "произвольные": "free",
}


def _normalize_allowed_angles(value: Any) -> str:
    key = str(value or "free").strip().lower()
    return _ALLOWED_MAP.get(key, "free")


def _normalize_border_mode(value: Any) -> str:
    t = str(value or "mark").strip().lower()
    if t in {"ignore", "игнорировать", "skip"}:
        return "ignore"
    if t in {"accept", "принимать"}:
        return "accept"
    return "mark"


def _normalize_hierarchy_mode(value: Any) -> bool:
    """True = external-only contours (RETR_EXTERNAL)."""
    t = str(value or "full").strip().lower()
    return t in {"external", "outer", "внешние"}


def metal_recovery_config_from_settings(settings: Any) -> MetalRecoveryConfig:
    max_w = getattr(settings, "metal_max_trace_width_px", None)
    if max_w is None or float(max_w) <= 0:
        max_w_px = None
    else:
        max_w_px = float(max_w)
    max_a = getattr(settings, "metal_max_area", None)
    if max_a is None or float(max_a) <= 0:
        max_area = None
    else:
        max_area = float(max_a)
    max_p = getattr(settings, "metal_max_perimeter", None)
    if max_p is None or float(max_p) <= 0:
        max_perimeter = None
    else:
        max_perimeter = float(max_p)

    gap = int(getattr(settings, "metal_gap_bridge_px", getattr(settings, "metal_morph_close_radius", 2)) or 2)
    speckle = int(
        getattr(settings, "metal_speckle_removal_px", getattr(settings, "metal_morph_open_radius", 0)) or 0
    )

    return MetalRecoveryConfig(
        noise_suppression=max(0, min(100, int(getattr(settings, "metal_noise_suppression", 20) or 20))),
        contrast_bias=max(-50.0, min(50.0, float(getattr(settings, "metal_contrast_bias", 0.0) or 0.0))),
        segmentation_strategy=normalize_metal_segmentation_strategy(
            getattr(settings, "metal_segmentation_strategy", "auto")
        ),
        gap_bridge_px=max(0, gap),
        speckle_removal_px=max(0, speckle),
        contour_smooth_px=max(0.0, float(getattr(settings, "metal_contour_smooth_px", 0.0) or 0.0)),
        morph_close_radius=max(0, gap),
        morph_open_radius=max(0, speckle),
        min_width_px=max(0.5, float(getattr(settings, "metal_min_trace_width_px", 8) or 8)),
        max_width_px=max_w_px,
        min_length_px=max(1.0, float(getattr(settings, "metal_min_trace_length_px", 8) or 8)),
        min_area=max(0.0, float(getattr(settings, "metal_min_area", 60) or 60)),
        max_area=max_area,
        min_perimeter=max(0.0, float(getattr(settings, "metal_min_perimeter", 32) or 32)),
        max_perimeter=max_perimeter,
        epsilon_simplify=max(0.0, float(getattr(settings, "epsilon", 2.0) or 2.0)),
        min_points=max(3, int(getattr(settings, "min_points", 4) or 4)),
        min_polygon_angle_deg=max(0.0, float(getattr(settings, "min_polygon_angle", 0.0) or 0.0)),
        approximation_enabled=bool(getattr(settings, "metal_approximation_enabled", True)),
        retrieval_external_only=_normalize_hierarchy_mode(getattr(settings, "metal_hierarchy_mode", "full")),
        allowed_angles=_normalize_allowed_angles(getattr(settings, "metal_allowed_angles", "free")),
        angle_tolerance_deg=max(0.5, float(getattr(settings, "metal_angle_tolerance_deg", 7.0) or 7.0)),
        min_straightness=max(0.05, min(1.0, float(getattr(settings, "metal_min_straightness", 0.2) or 0.2))),
        allow_t_junction=bool(getattr(settings, "metal_allow_t_junction", True)),
        border_mode=_normalize_border_mode(getattr(settings, "metal_border_handling", "mark")),
        check_contour_validity=bool(getattr(settings, "metal_check_contour_validity", True)),
        min_inner_hole_area=max(0.0, float(getattr(settings, "min_inner_hole_area", 100.0) or 100.0)),
        preset_name=str(getattr(settings, "metal_preset", "standard") or "standard"),
        use_wide_conductor_gradient=bool(getattr(settings, "metal_use_wide_conductor_gradient", False)),
        wide_gradient_profile_radius_px=max(
            1, int(getattr(settings, "metal_wide_gradient_profile_radius_px", 8) or 8)
        ),
        wide_gradient_min_direction_confidence=max(
            0.0,
            min(
                1.0,
                float(getattr(settings, "metal_wide_gradient_min_direction_confidence", 0.15) or 0.15),
            ),
        ),
        wide_gradient_min_pair_length_px=max(
            4.0, float(getattr(settings, "metal_wide_gradient_min_pair_length_px", 24.0) or 24.0)
        ),
        wide_gradient_parallel_tolerance_deg=max(
            0.5,
            float(getattr(settings, "metal_wide_gradient_parallel_tolerance_deg", 10.0) or 10.0),
        ),
        wide_gradient_max_edge_gap_px=max(
            0, int(getattr(settings, "metal_wide_gradient_max_edge_gap_px", 5) or 5)
        ),
        wide_gradient_min_overlap_ratio=max(
            0.05,
            min(1.0, float(getattr(settings, "metal_wide_gradient_min_overlap_ratio", 0.5) or 0.5)),
        ),
        edge_close_cap_px=max(5, int(getattr(settings, "metal_edge_close_cap_px", 9) or 9)),
        edge_watershed_split=bool(getattr(settings, "metal_edge_watershed_split", True)),
        edge_watershed_dist_peak_frac=max(
            0.1,
            min(0.9, float(getattr(settings, "metal_edge_watershed_dist_peak_frac", 0.38) or 0.38)),
        ),
        edge_watershed_max_pixels=int(getattr(settings, "metal_edge_watershed_max_pixels", 3_000_000) or 0)
        or None,
    )
