"""detector: contour extraction settings metal."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from ....application.preview_cancellation import raise_if_preview_cancelled
from ....application.processing import ContourExtractionSettings
from contour.domain import PolygonData
from contour.utils import ensure_binary_mask
from contour.vision.metal_recovery.gradient_watershed import (
    gradient_watershed_config_from_object,
)
from contour.vision.metal_recovery.pipeline_stages import build_metal_segmentation_mask_staged
from contour.vision.metal_recovery.segmentation import (
    MetalSegmentationConfig,
    normalize_metal_adaptive_method,
    resolve_metal_segmentation_strategy,
)

from contour.vision.metal_recovery.detector_parts.module_state import _CONTOUR_CACHE

@dataclass(slots=True)
class MetalPolygonRecord:
    polygon: PolygonData
    area: float = 0.0
    perimeter: float = 0.0
    border_touch: bool = False
    reject_reason: str = ""

@dataclass(slots=True)
class MetalDetectionResult:
    accepted: list[PolygonData] = field(default_factory=list)
    rejected: list[MetalPolygonRecord] = field(default_factory=list)
    suspicious: list[MetalPolygonRecord] = field(default_factory=list)
    border: list[MetalPolygonRecord] = field(default_factory=list)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    params_snapshot: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class _ContourStageCache:
    mask_sig: str = ""
    config_sig: tuple[Any, ...] = ()
    polygons: list[PolygonData] = field(default_factory=list)

def clear_metal_contour_cache() -> None:
    _CONTOUR_CACHE.clear()

def _requested_segmentation_strategy(config: MetalRecoveryConfig) -> str:
    return resolve_metal_segmentation_strategy(
        config.segmentation_strategy,
        use_wide_conductor_gradient=bool(config.use_wide_conductor_gradient),
    )

def _segmentation_config_from_recovery(config: MetalRecoveryConfig) -> MetalSegmentationConfig:
    watershed = gradient_watershed_config_from_object(config)
    return MetalSegmentationConfig(
        min_contrast=max(1.0, min(255.0, float(config.min_contrast))),
        gap_bridge_px=max(0, int(config.gap_bridge_px)),
        speckle_removal_px=max(0, int(config.speckle_removal_px)),
        min_component_area=max(0, int(config.min_component_area or config.min_area)),
        segmentation_strategy=_requested_segmentation_strategy(config),
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
        structural_variant=str(getattr(config, "structural_variant", "s2") or "s2"),
        adaptive_block_size=max(0, min(255, int(config.adaptive_block_size))),
        adaptive_c=max(-64.0, min(64.0, float(config.adaptive_c))),
        adaptive_method=normalize_metal_adaptive_method(config.adaptive_method),
        strategy_parameters=config.strategy_configs.to_dict(),
    )

def _contour_config_signature(config: MetalRecoveryConfig) -> tuple[Any, ...]:
    return (
        float(config.epsilon_simplify) if config.approximation_enabled else 0.0,
        bool(config.approximation_enabled),
        str(config.approximation_mode or "CHAIN_APPROX_SIMPLE"),
        bool(config.retrieval_external_only),
        str(config.retrieval_mode or "RETR_TREE"),
        float(config.min_area),
        config.max_area,
        float(config.min_perimeter),
        config.max_perimeter,
        float(config.min_width_px),
        config.max_width_px,
        float(config.min_length_px),
        max(3, int(config.min_points)),
        float(config.min_polygon_angle_deg),
        float(config.min_inner_hole_area),
        str(config.border_mode or "mark"),
    )

def build_metal_extraction_mask(
    gray: np.ndarray,
    config: MetalRecoveryConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pipeline: filtered gray → Otsu → morphology → binary mask."""
    raise_if_preview_cancelled()
    result = build_metal_segmentation_mask_staged(gray, _segmentation_config_from_recovery(config))
    return ensure_binary_mask(result.mask), dict(result.debug_images)

def contour_extraction_settings_from_metal(config: MetalRecoveryConfig) -> ContourExtractionSettings:
    """Map recognition settings to ``extract_polygons`` (findContours + filters)."""
    retrieval = (
        "RETR_EXTERNAL"
        if config.retrieval_external_only
        else str(config.retrieval_mode or "RETR_TREE")
    )
    max_area = config.max_area if config.max_area is not None and config.max_area > 0 else None
    max_perimeter = (
        config.max_perimeter if config.max_perimeter is not None and config.max_perimeter > 0 else None
    )
    return ContourExtractionSettings(
        object_type="conductor",
        output_mode="polygon",
        extraction_profile="conductors",
        retrieval_mode=retrieval,
        approximation_mode=str(config.approximation_mode or "CHAIN_APPROX_SIMPLE"),
        epsilon=float(config.epsilon_simplify) if config.approximation_enabled else 0.0,
        min_area=float(config.min_area),
        max_area=max_area,
        min_perimeter=float(config.min_perimeter),
        max_perimeter=max_perimeter,
        min_polygon_width_px=float(config.min_width_px),
        min_points=max(3, int(config.min_points)),
        min_polygon_angle=float(config.min_polygon_angle_deg),
        min_inner_hole_area=float(config.min_inner_hole_area),
        exclude_border_touching=(str(config.border_mode).strip().lower() == "ignore"),
    )

def _border_touch_points(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    margin_px: int = 1,
) -> bool:
    limit_x = max(0, int(width) - 1)
    limit_y = max(0, int(height) - 1)
    margin = max(0, int(margin_px))
    for x_coord, y_coord in points:
        if (
            x_coord <= margin
            or y_coord <= margin
            or x_coord >= limit_x - margin
            or y_coord >= limit_y - margin
        ):
            return True
    return False

def _trace_length_px(polygon: PolygonData) -> float:
    _, _, width, height = polygon.bbox
    return max(float(width), float(height))

def _trace_width_px(polygon: PolygonData) -> float:
    _, _, width, height = polygon.bbox
    return min(float(width), float(height))

def _passes_trace_geometry(polygon: PolygonData, config: MetalRecoveryConfig) -> bool:
    if polygon.is_hole:
        return True
    if config.min_length_px > 0.0 and _trace_length_px(polygon) < float(config.min_length_px):
        return False
    if config.max_width_px is None or config.max_width_px <= 0.0:
        return True
    return _trace_width_px(polygon) <= float(config.max_width_px)
