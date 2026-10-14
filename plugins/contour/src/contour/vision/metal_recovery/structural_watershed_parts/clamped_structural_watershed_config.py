"""structural_watershed: clamped structural watershed config."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

def normalize_structural_variant(value: Any) -> str:
    text = str(value or "s2").strip().lower().replace("-", "_")
    aliases = {
        "s1": "s1",
        "existing_markers_gradient": "s1",
        "s2": "s2",
        "ridge_interior_gradient": "s2",
        "s3": "s3",
        "ridge_interior_oriented": "s3",
        "s4": "s4",
        "multiscale_oriented": "s4",
        "s5": "s5",
        "multilabel_gradient": "s5",
        "s6": "s6",
        "multilabel_oriented": "s6",
        "s7": "s7",
        "geodesic_competition": "s7",
        "s8": "s8",
        "ridge_link_geodesic": "s8",
        "s9": "s9",
        "wide_consolidate_geodesic": "s9",
        "s10": "s10",
        "marker_consolidate_geodesic": "s10",
        "s11": "s11",
        "separator_combine_geodesic": "s11",
        "s12": "s12",
        "oriented_veto_geodesic": "s12",
        "s13": "s13",
        "conductor_band_geodesic": "s13",
        "s14": "s14",
        "band_identity_geodesic": "s14",
    }
    return aliases.get(text, "s2")

@dataclass(frozen=True, slots=True)
class StructuralWatershedConfig:
    """Physically named controls for structural marker watershed."""

    variant: str = "s2"
    smoothing_sigma: float = 1.0
    ridge_scale_min: float = 1.0
    ridge_scale_max: float = 6.0
    min_ridge_confidence: float = 0.22
    min_orientation_coherence: float = 0.45
    orientation_smoothing_scale: float = 3.0
    min_marker_length: int = 10
    min_marker_area: int = 16
    gap_link_px: int = 2
    wide_interior_radius: float = 4.0
    boundary_gradient_weight: float = 1.0
    boundary_continuity_weight: float = 0.6
    boundary_rim_weight: float = 0.8
    boundary_orientation_weight: float = 0.4

def clamped_structural_watershed_config(
    *,
    variant: str = "s2",
    smoothing_sigma: float = 1.0,
    ridge_scale_min: float = 1.0,
    ridge_scale_max: float = 6.0,
    min_ridge_confidence: float = 0.22,
    min_orientation_coherence: float = 0.45,
    orientation_smoothing_scale: float = 3.0,
    min_marker_length: int = 10,
    min_marker_area: int = 16,
    gap_link_px: int = 2,
    wide_interior_radius: float = 4.0,
    boundary_gradient_weight: float = 1.0,
    boundary_continuity_weight: float = 0.6,
    boundary_rim_weight: float = 0.8,
    boundary_orientation_weight: float = 0.4,
) -> StructuralWatershedConfig:
    scale_min = max(0.5, min(8.0, float(ridge_scale_min)))
    scale_max = max(scale_min, min(16.0, float(ridge_scale_max)))
    return StructuralWatershedConfig(
        variant=normalize_structural_variant(variant),
        smoothing_sigma=max(0.1, min(8.0, float(smoothing_sigma))),
        ridge_scale_min=scale_min,
        ridge_scale_max=scale_max,
        min_ridge_confidence=max(0.02, min(0.8, float(min_ridge_confidence))),
        min_orientation_coherence=max(0.0, min(1.0, float(min_orientation_coherence))),
        orientation_smoothing_scale=max(1.0, min(16.0, float(orientation_smoothing_scale))),
        min_marker_length=max(2, min(64, int(min_marker_length))),
        min_marker_area=max(2, min(256, int(min_marker_area))),
        gap_link_px=max(0, min(12, int(gap_link_px))),
        wide_interior_radius=max(1.0, min(24.0, float(wide_interior_radius))),
        boundary_gradient_weight=max(0.0, min(4.0, float(boundary_gradient_weight))),
        boundary_continuity_weight=max(0.0, min(4.0, float(boundary_continuity_weight))),
        boundary_rim_weight=max(0.0, min(4.0, float(boundary_rim_weight))),
        boundary_orientation_weight=max(0.0, min(4.0, float(boundary_orientation_weight))),
    )
