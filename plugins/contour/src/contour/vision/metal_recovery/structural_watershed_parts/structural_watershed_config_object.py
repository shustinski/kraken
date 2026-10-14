"""structural_watershed: structural watershed config object."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from contour.vision.metal_recovery.gradient_watershed import (
    GradientWatershedConfig,
)
from contour.vision.metal_recovery.marker_consolidation import (
    MarkerConsolidationStats,
)

from contour.vision.metal_recovery.structural_watershed_parts.clamped_structural_watershed_config import clamped_structural_watershed_config

from contour.vision.metal_recovery.structural_watershed_parts.run_structural_watershed import run_structural_watershed

def structural_watershed_config_from_object(source: object) -> StructuralWatershedConfig:
    return clamped_structural_watershed_config(
        variant=str(getattr(source, "structural_variant", "s2") or "s2"),
        smoothing_sigma=float(
            getattr(
                source,
                "watershed_smoothing_sigma",
                getattr(source, "smoothing_sigma", 1.0),
            )
            or 1.0
        ),
        ridge_scale_min=float(getattr(source, "structural_ridge_scale_min", 1.0) or 1.0),
        ridge_scale_max=float(getattr(source, "structural_ridge_scale_max", 6.0) or 6.0),
        min_ridge_confidence=float(getattr(source, "structural_min_ridge_confidence", 0.22) or 0.22),
        min_orientation_coherence=float(
            getattr(source, "structural_min_orientation_coherence", 0.45) or 0.45
        ),
        orientation_smoothing_scale=float(
            getattr(source, "structural_orientation_smoothing_scale", 3.0) or 3.0
        ),
        min_marker_length=int(getattr(source, "structural_min_marker_length", 10) or 10),
        min_marker_area=int(getattr(source, "structural_min_marker_area", 16) or 16),
        gap_link_px=int(getattr(source, "structural_gap_link_px", 2) or 0),
        wide_interior_radius=float(getattr(source, "structural_wide_interior_radius", 4.0) or 4.0),
        boundary_gradient_weight=float(
            getattr(source, "structural_boundary_gradient_weight", 1.0) or 0.0
        ),
        boundary_continuity_weight=float(
            getattr(source, "structural_boundary_continuity_weight", 0.6) or 0.0
        ),
        boundary_rim_weight=float(getattr(source, "structural_boundary_rim_weight", 0.8) or 0.0),
        boundary_orientation_weight=float(
            getattr(source, "structural_boundary_orientation_weight", 0.4) or 0.0
        ),
    )

@dataclass(frozen=True, slots=True)
class StructuralWatershedResult:
    mask: np.ndarray
    instance_labels: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    instance_count: int = 0
    label_fragment_count: int = 0
    consolidation: MarkerConsolidationStats = field(default_factory=MarkerConsolidationStats)

def structural_watershed_mask(
    gray: np.ndarray,
    watershed_config: GradientWatershedConfig | None = None,
    structural_config: StructuralWatershedConfig | None = None,
    *,
    check_presence: bool = True,
) -> np.ndarray:
    return run_structural_watershed(
        gray,
        watershed_config,
        structural_config,
        check_presence=check_presence,
    ).mask
