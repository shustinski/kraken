from __future__ import annotations

from typing import Any

from .detector import MetalRecoveryConfig


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

    external_only = _normalize_hierarchy_mode(getattr(settings, "metal_hierarchy_mode", "full"))
    return MetalRecoveryConfig(
        contrast_bias=max(-50.0, min(50.0, float(getattr(settings, "metal_contrast_bias", 0.0) or 0.0))),
        segmentation_strategy="legacy_otsu",
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
        retrieval_mode="RETR_EXTERNAL" if external_only else str(getattr(settings, "retrieval_mode", "RETR_TREE")),
        approximation_mode=str(getattr(settings, "approximation_mode", "CHAIN_APPROX_SIMPLE")),
        border_mode=_normalize_border_mode(getattr(settings, "metal_border_handling", "mark")),
        min_inner_hole_area=max(0.0, float(getattr(settings, "min_inner_hole_area", 100.0) or 100.0)),
        min_component_area=max(
            0.0,
            float(getattr(settings, "metal_min_object_area", getattr(settings, "metal_min_area", 60)) or 60),
        ),
        preset_name=str(getattr(settings, "metal_preset", "standard") or "standard"),
    )
