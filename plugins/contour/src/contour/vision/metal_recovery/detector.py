"""Conductor recognition: segmentation mask → findContours + post-filters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import ceil
from typing import Any

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ...application.processing import ContourExtractionSettings
from ...contour_extractor import extract_polygons
from ...domain import PolygonData
from ...utils import ensure_binary_mask, ensure_uint8
from .gradient_watershed import (
    analyze_metal_presence,
    build_conductor_seeds,
    gradient_watershed_config_from_object,
)
from .pipeline_stages import axis_gradient_debug_images, build_metal_segmentation_mask_staged, image_signature
from .segmentation import (
    MetalSegmentationConfig,
    normalize_metal_adaptive_method,
    normalize_metal_segmentation_strategy,
    resolve_metal_segmentation_strategy,
)
from .strategy_registry import IMPLEMENTED_NEW_STRATEGIES, MetalStrategyConfigs


def _normalize_metal_extraction_mode(value: Any) -> str:
    """Compatibility bridge to the canonical segmentation normalizer."""
    return normalize_metal_segmentation_strategy(value)


@dataclass(slots=True)
class MetalRecoveryConfig:
    """Segmentation + contour extraction parameters exposed in the UI."""

    min_contrast: float = 50.0
    min_object_source_contrast: float = 12.0
    min_object_rim_contrast: float = 36.0
    min_object_rim_area_fraction: float = 0.001
    contrast_bias: float = 0.0  # Deprecated compatibility field; ignored by segmentation.
    min_hole_source_contrast: float = 8.0
    min_hole_source_contrast_fraction: float = 0.35
    gap_bridge_px: int = 2
    speckle_removal_px: int = 0
    min_width_px: float = 8.0
    max_width_px: float | None = None
    min_length_px: float = 8.0
    min_area: float = 60.0
    max_area: float | None = None
    min_perimeter: float = 32.0
    max_perimeter: float | None = None
    epsilon_simplify: float = 2.0
    min_points: int = 4
    min_polygon_angle_deg: float = 0.0
    approximation_enabled: bool = True
    retrieval_external_only: bool = False
    border_mode: str = "mark"
    min_inner_hole_area: float = 100.0
    min_component_area: float = 60.0
    preset_name: str = "standard"
    retrieval_mode: str = "RETR_TREE"
    approximation_mode: str = "CHAIN_APPROX_SIMPLE"
    # Legacy / ignored at runtime (filters tab handles preprocessing).
    noise_suppression: int = 0
    segmentation_strategy: str = "legacy_otsu"
    auto_contrast_step: float = 10.0
    auto_source_contrast_step: float = 4.0
    auto_directional_gap_bridge_px: int = 3
    auto_directional_gap_min_source_intensity: float = 45.0
    use_wide_conductor_gradient: bool = False
    watershed_smoothing_sigma: float = 1.0
    watershed_core_margin: float = 8.0
    watershed_groove_margin: float = 16.0
    watershed_rim_probe_px: int = 6
    watershed_seed_speckle_px: int = 4
    watershed_valley_span_px: int = 5
    watershed_valley_depth: float = 45.0
    adaptive_block_size: int = 0
    adaptive_c: float = 0.0
    adaptive_method: str = "gaussian"
    random_walker_beta: float = 90.0
    random_walker_iterations: int = 160
    graph_cut_iterations: int = 5
    reconstruction_erode_px: int = 0
    boundary_relief: float = 16.0
    boundary_background_sigma: float = 12.0
    structural_variant: str = "s2"
    strategy_configs: MetalStrategyConfigs = field(default_factory=MetalStrategyConfigs)

    def __post_init__(self) -> None:
        if isinstance(self.strategy_configs, MetalStrategyConfigs):
            return
        if isinstance(self.strategy_configs, Mapping):
            self.strategy_configs = MetalStrategyConfigs.from_mapping(self.strategy_configs)
            return
        raise TypeError("strategy_configs must be MetalStrategyConfigs or a mapping")

    def to_snapshot(self) -> dict[str, Any]:
        snapshot = {
            field.name: getattr(self, field.name)
            for field in MetalRecoveryConfig.__dataclass_fields__.values()
        }
        snapshot["strategy_configs"] = self.strategy_configs.to_dict()
        return snapshot


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


_CONTOUR_CACHE: dict[str, _ContourStageCache] = {}
_CONTOUR_CACHE_MAX_ITEMS = 8


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


def _extract_polygons_per_instance(
    labels: np.ndarray,
    config: MetalRecoveryConfig,
) -> list[PolygonData]:
    """Extract each instance in its bounding box without full-frame rescans."""

    extraction_settings = contour_extraction_settings_from_metal(config)
    polygons: list[PolygonData] = []
    next_id = 1
    active_labels = np.unique(labels)
    active_labels = active_labels[active_labels > 0].astype(np.int32)
    if active_labels.size == 0:
        return polygons
    height, width = labels.shape
    maximum_label = int(active_labels.max())
    min_x = np.full(maximum_label + 1, width, dtype=np.int32)
    max_x = np.full(maximum_label + 1, -1, dtype=np.int32)
    min_y = np.full(maximum_label + 1, height, dtype=np.int32)
    max_y = np.full(maximum_label + 1, -1, dtype=np.int32)
    x_coordinates = np.broadcast_to(np.arange(width, dtype=np.int32), labels.shape)
    y_coordinates = np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], labels.shape)
    np.minimum.at(min_x, labels, x_coordinates)
    np.maximum.at(max_x, labels, x_coordinates)
    np.minimum.at(min_y, labels, y_coordinates)
    np.maximum.at(max_y, labels, y_coordinates)
    for label_id in active_labels:
        x0 = int(min_x[label_id])
        x1 = int(max_x[label_id]) + 1
        y0 = int(min_y[label_id])
        y1 = int(max_y[label_id]) + 1
        if x1 <= x0 or y1 <= y0:
            continue
        instance_mask = np.where(labels[y0:y1, x0:x1] == label_id, 255, 0).astype(np.uint8)
        instance_polygons = extract_polygons(instance_mask, extraction_settings)
        id_map = {
            int(polygon.id): next_id + index
            for index, polygon in enumerate(instance_polygons)
        }
        for polygon in instance_polygons:
            old_parent = polygon.parent_id
            polygon.id = id_map[int(polygon.id)]
            polygon.parent_id = None if old_parent is None else id_map.get(int(old_parent))
            polygon.points = [
                (float(x_coord) + x0, float(y_coord) + y0)
                for x_coord, y_coord in polygon.points
            ]
            bbox_x, bbox_y, bbox_width, bbox_height = polygon.bbox
            polygon.bbox = (bbox_x + x0, bbox_y + y0, bbox_width, bbox_height)
            polygons.append(polygon)
        next_id += len(instance_polygons)
    return polygons


def _extract_polygons_cached(mask: np.ndarray, config: MetalRecoveryConfig) -> list[PolygonData]:
    mask_sig = image_signature(ensure_uint8(mask))
    cfg_sig = _contour_config_signature(config)
    cached = _CONTOUR_CACHE.get(mask_sig)
    if cached is not None and cached.mask_sig == mask_sig and cached.config_sig == cfg_sig:
        return [polygon.clone() for polygon in cached.polygons]

    extraction_settings = contour_extraction_settings_from_metal(config)
    polygons = extract_polygons(mask, extraction_settings)
    while len(_CONTOUR_CACHE) >= _CONTOUR_CACHE_MAX_ITEMS:
        oldest = next(iter(_CONTOUR_CACHE))
        _CONTOUR_CACHE.pop(oldest, None)
    _CONTOUR_CACHE[mask_sig] = _ContourStageCache(
        mask_sig=mask_sig,
        config_sig=cfg_sig,
        polygons=[polygon.clone() for polygon in polygons],
    )
    return polygons


def _renumber_polygons_preserving_parents(polygons: list[PolygonData]) -> None:
    if not polygons:
        return
    old_to_new: dict[int, int] = {}
    for new_id, poly in enumerate(polygons, start=1):
        old_to_new[int(poly.id)] = new_id
    for poly in polygons:
        poly.id = old_to_new[int(poly.id)]
    for poly in polygons:
        if poly.parent_id is not None:
            poly.parent_id = old_to_new.get(int(poly.parent_id), poly.parent_id)


def _polygon_interior_median(
    gray: np.ndarray,
    polygon: PolygonData,
    *,
    excluded: list[PolygonData] | None = None,
) -> float | None:
    points = np.asarray(polygon.points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    height, width = gray.shape[:2]
    x_min = max(0, int(np.floor(points[:, 0].min())))
    y_min = max(0, int(np.floor(points[:, 1].min())))
    x_max = min(width - 1, int(np.ceil(points[:, 0].max())))
    y_max = min(height - 1, int(np.ceil(points[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        return None

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)

    def _local_points(item: PolygonData) -> np.ndarray:
        item_points = np.asarray(item.points, dtype=np.float32).reshape(-1, 2)
        item_points[:, 0] -= x_min
        item_points[:, 1] -= y_min
        return np.rint(item_points).astype(np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(mask, [_local_points(polygon)], 255)
    for item in excluded or []:
        if len(item.points) >= 3:
            cv2.fillPoly(mask, [_local_points(item)], 0)
    values = gray[y_min : y_max + 1, x_min : x_max + 1][mask > 0]
    if values.size == 0:
        return None
    return float(np.median(values))


def _filter_holes_by_source_contrast(
    polygons: list[PolygonData],
    source_image: np.ndarray | None,
    *,
    min_source_contrast: float,
    min_source_contrast_fraction: float,
) -> list[PolygonData]:
    if source_image is None or not any(polygon.is_hole for polygon in polygons):
        return polygons
    if source_image.ndim == 3:
        source_gray = cv2.cvtColor(ensure_uint8(source_image), cv2.COLOR_BGR2GRAY)
    else:
        source_gray = ensure_uint8(source_image)
    if source_gray.size == 0:
        return polygons

    otsu_threshold, _binary = cv2.threshold(
        source_gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    lower_values = source_gray[source_gray <= otsu_threshold]
    upper_values = source_gray[source_gray > otsu_threshold]
    if lower_values.size == 0 or upper_values.size == 0:
        return polygons
    source_class_separation = float(np.median(upper_values) - np.median(lower_values))
    if source_class_separation <= 0.0:
        return polygons
    min_contrast = max(
        max(0.0, float(min_source_contrast)),
        source_class_separation * max(0.0, min(1.0, float(min_source_contrast_fraction))),
    )

    polygons_by_id = {int(polygon.id): polygon for polygon in polygons}
    holes_by_parent: dict[int, list[PolygonData]] = {}
    for polygon in polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)

    parent_medians: dict[int, float | None] = {}
    kept: list[PolygonData] = []
    for polygon in polygons:
        if not polygon.is_hole:
            kept.append(polygon)
            continue
        if polygon.parent_id is None:
            continue
        parent_id = int(polygon.parent_id)
        parent = polygons_by_id.get(parent_id)
        if parent is None or parent.is_hole:
            continue
        if parent_id not in parent_medians:
            parent_medians[parent_id] = _polygon_interior_median(
                source_gray,
                parent,
                excluded=holes_by_parent.get(parent_id),
            )
        parent_median = parent_medians[parent_id]
        hole_median = _polygon_interior_median(source_gray, polygon)
        if parent_median is None or hole_median is None:
            continue
        if hole_median <= float(otsu_threshold) and parent_median - hole_median >= min_contrast:
            kept.append(polygon)
    return kept


def _filter_conductors_by_source_contrast(
    polygons: list[PolygonData],
    source_image: np.ndarray | None,
    *,
    min_contrast: float,
    min_rim_contrast: float,
    min_rim_area_fraction: float,
) -> list[PolygonData]:
    """Reject dark shadow islands that watershed can grow beside a conductor."""
    if source_image is None or min_contrast <= 0.0 or not polygons:
        return polygons
    if source_image.ndim == 3:
        source_gray = cv2.cvtColor(ensure_uint8(source_image), cv2.COLOR_BGR2GRAY)
    else:
        source_gray = ensure_uint8(source_image)

    by_parent: dict[int, list[PolygonData]] = {}
    for polygon in polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            by_parent.setdefault(int(polygon.parent_id), []).append(polygon)

    kept_parent_ids: set[int] = set()
    ring_radius = 10
    min_rim_backed_area_fraction = max(0.000001, min(1.0, float(min_rim_area_fraction)))
    required_rim_contrast = max(0.0, float(min_rim_contrast))
    rim_percentile = 90.0
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1),
    )
    for polygon in polygons:
        if polygon.is_hole:
            continue
        points = np.asarray(polygon.points, dtype=np.int32).reshape(-1, 2)
        x_min = max(0, int(points[:, 0].min()) - ring_radius)
        x_max = min(source_gray.shape[1] - 1, int(points[:, 0].max()) + ring_radius)
        y_min = max(0, int(points[:, 1].min()) - ring_radius)
        y_max = min(source_gray.shape[0] - 1, int(points[:, 1].max()) + ring_radius)
        local_source = source_gray[y_min : y_max + 1, x_min : x_max + 1]

        filled = np.zeros(local_source.shape, dtype=np.uint8)
        local_polygon_points = points.copy()
        local_polygon_points[:, 0] -= x_min
        local_polygon_points[:, 1] -= y_min
        cv2.fillPoly(filled, [local_polygon_points.reshape(-1, 1, 2)], 255)
        interior = filled.copy()
        for hole in by_parent.get(int(polygon.id), ()):
            local_hole_points = np.asarray(hole.points, dtype=np.int32).reshape(-1, 2).copy()
            local_hole_points[:, 0] -= x_min
            local_hole_points[:, 1] -= y_min
            cv2.fillPoly(interior, [local_hole_points.reshape(-1, 1, 2)], 0)
        inside_values = local_source[interior > 0]
        ring = (cv2.dilate(filled, ring_kernel) > 0) & (filled == 0)
        outside_values = local_source[ring]
        if inside_values.size == 0 or outside_values.size == 0:
            kept_parent_ids.add(int(polygon.id))
            continue
        inside_median = float(np.median(inside_values))
        outside_median = float(np.median(outside_values))
        interior_contrast = inside_median - outside_median
        area_fraction = float(inside_values.size / max(1, source_gray.size))
        inner_edge = (interior > 0) & (
            cv2.erode(interior, np.ones((3, 3), dtype=np.uint8)) == 0
        )
        near_ring = (cv2.dilate(filled, np.ones((3, 3), dtype=np.uint8)) > 0) & (
            filled == 0
        )
        edge_values = local_source[inner_edge]
        near_values = local_source[near_ring]
        rim_contrast = (
            float(np.percentile(edge_values, rim_percentile) - np.median(near_values))
            if edge_values.size > 0 and near_values.size > 0
            else 0.0
        )

        if interior_contrast >= float(min_contrast):
            is_sizeable_component = area_fraction >= min_rim_backed_area_fraction
            has_strong_interior = interior_contrast >= 2.0 * float(min_contrast)
            has_strong_rim = rim_contrast >= required_rim_contrast
            if is_sizeable_component or has_strong_interior or has_strong_rim:
                kept_parent_ids.add(int(polygon.id))
            continue

        # Wide SEM conductors can have a substrate-like interior while their
        # bright material rim remains unambiguous.  Use that evidence only for
        # dominant candidates so textured background islands are not promoted.
        is_large_rim_candidate = (
            area_fraction >= 6.0 * min_rim_backed_area_fraction
            and interior_contrast >= 0.75 * float(min_contrast)
            and rim_contrast >= 0.75 * required_rim_contrast
        )
        is_dominant_component = area_fraction >= 50.0 * min_rim_backed_area_fraction
        if is_large_rim_candidate or (
            rim_contrast >= (2.0 / 3.0) * required_rim_contrast and is_dominant_component
        ):
            kept_parent_ids.add(int(polygon.id))

    return [
        polygon
        for polygon in polygons
        if (
            int(polygon.parent_id) in kept_parent_ids
            if polygon.is_hole and polygon.parent_id is not None
            else not polygon.is_hole and int(polygon.id) in kept_parent_ids
        )
    ]


def _rasterize_accepted_conductors(
    polygons: list[PolygonData],
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a filled mask while preserving holes and nested conductors."""
    holes_by_parent: dict[int, list[PolygonData]] = {}
    conductors: list[tuple[float, PolygonData]] = []
    for polygon in polygons:
        if polygon.is_hole:
            if polygon.parent_id is not None:
                holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)
            continue
        conductors.append((abs(float(polygon.area)), polygon))

    mask = np.zeros(shape, dtype=np.uint8)
    for _area, conductor in sorted(conductors, key=lambda item: item[0], reverse=True):
        points = np.asarray(conductor.points, dtype=np.int32).reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        x_min = max(0, int(points[:, 0].min()))
        x_max = min(shape[1] - 1, int(points[:, 0].max()))
        y_min = max(0, int(points[:, 1].min()))
        y_max = min(shape[0] - 1, int(points[:, 1].max()))
        if x_max < x_min or y_max < y_min:
            continue
        region = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        local_points = points.copy()
        local_points[:, 0] -= x_min
        local_points[:, 1] -= y_min
        cv2.fillPoly(region, [local_points.reshape(-1, 1, 2)], 255)
        for hole in holes_by_parent.get(int(conductor.id), ()):
            hole_points = np.asarray(hole.points, dtype=np.int32).reshape(-1, 2)
            if hole_points.shape[0] < 3:
                continue
            local_hole_points = hole_points.copy()
            local_hole_points[:, 0] -= x_min
            local_hole_points[:, 1] -= y_min
            cv2.fillPoly(region, [local_hole_points.reshape(-1, 1, 2)], 0)
        target = mask[y_min : y_max + 1, x_min : x_max + 1]
        target[region > 0] = 255
    return mask


def _detect_metalization_explicit(
    image: np.ndarray,
    config: MetalRecoveryConfig,
    *,
    source_image: np.ndarray | None = None,
    mask_override: np.ndarray | None = None,
    include_axis_debug: bool = True,
) -> MetalDetectionResult:
    if image.ndim == 3:
        gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2GRAY)
    else:
        gray = ensure_uint8(image)
    if gray.size == 0:
        return MetalDetectionResult(params_snapshot=config.to_snapshot())

    if mask_override is None:
        segmentation = build_metal_segmentation_mask_staged(
            gray,
            _segmentation_config_from_recovery(config),
        )
        mask = ensure_binary_mask(segmentation.mask)
        pre_dbg = dict(segmentation.debug_images)
        instance_labels = segmentation.instance_labels
        strategy_timings = dict(segmentation.timings_ms)
        strategy_debug_data = dict(segmentation.debug_data)
    else:
        mask = ensure_binary_mask(mask_override)
        pre_dbg = {}
        instance_labels = None
        strategy_timings = {}
        strategy_debug_data = {}
    raise_if_preview_cancelled()

    h, w = mask.shape[:2]
    hole_source = source_image
    if hole_source is not None and hole_source.shape[:2] != gray.shape[:2]:
        hole_source = None
    from .structural_watershed import INSTANCE_IDENTITY_VARIANTS

    if (
        instance_labels is not None
        and (
            _requested_segmentation_strategy(config) in IMPLEMENTED_NEW_STRATEGIES
            or str(getattr(config, "structural_variant", "s2") or "s2") in INSTANCE_IDENTITY_VARIANTS
        )
        and np.any(instance_labels > 0)
    ):
        polygons = _extract_polygons_per_instance(instance_labels, config)
    else:
        polygons = _extract_polygons_cached(mask, config)
    polygons = _filter_holes_by_source_contrast(
        polygons,
        hole_source,
        min_source_contrast=config.min_hole_source_contrast,
        min_source_contrast_fraction=config.min_hole_source_contrast_fraction,
    )
    polygons = _filter_conductors_by_source_contrast(
        polygons,
        hole_source,
        min_contrast=config.min_object_source_contrast,
        min_rim_contrast=config.min_object_rim_contrast,
        min_rim_area_fraction=config.min_object_rim_area_fraction,
    )

    accepted: list[PolygonData] = []
    border_mode = str(config.border_mode or "mark").strip().lower()

    for polygon in polygons:
        if not _passes_trace_geometry(polygon, config):
            continue

        poly = polygon.clone()
        if polygon.is_hole:
            poly.category = "conductor"
            accepted.append(poly)
            continue

        touches_border = _border_touch_points(poly.points, width=w, height=h)
        if border_mode == "ignore" and touches_border:
            continue

        poly.category = "conductor"
        if border_mode == "mark" and touches_border:
            poly.category = "metal_border"

        accepted.append(poly)

    accepted_conductor_ids = {int(polygon.id) for polygon in accepted if not polygon.is_hole}
    accepted = [
        polygon
        for polygon in accepted
        if not polygon.is_hole
        or (polygon.parent_id is not None and int(polygon.parent_id) in accepted_conductor_ids)
    ]
    _renumber_polygons_preserving_parents(accepted)
    border = [
        MetalPolygonRecord(
            polygon=polygon.clone(),
            area=float(polygon.area),
            perimeter=float(polygon.perimeter),
            border_touch=True,
        )
        for polygon in accepted
        if polygon.category == "metal_border"
    ]
    accepted_mask = _rasterize_accepted_conductors(accepted, mask.shape[:2])

    raw_contours, _hierarchy = cv2.findContours(
        ensure_binary_mask(mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    dbg: dict[str, np.ndarray] = {
        "metal_source_gray": gray,
        "metal_binary_mask": mask,
        "metal_filtered_mask": accepted_mask,
        "metal_contour_extraction_mask": mask,
    }
    for key, value in pre_dbg.items():
        if isinstance(value, np.ndarray):
            dbg[key] = value
    if include_axis_debug:
        dbg.update(axis_gradient_debug_images(gray))
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if raw_contours:
        cv2.drawContours(vis, raw_contours, -1, (0, 255, 0), 1)
    dbg["metal_contours_raw"] = vis
    dbg["metal_width_check"] = cv2.cvtColor(accepted_mask, cv2.COLOR_GRAY2BGR)

    params_snapshot = config.to_snapshot()
    if strategy_timings:
        params_snapshot["strategy_timings_ms"] = strategy_timings
    if strategy_debug_data:
        params_snapshot["strategy_debug_data"] = strategy_debug_data

    return MetalDetectionResult(
        accepted=accepted,
        rejected=[],
        suspicious=[],
        border=border,
        debug_images=dbg,
        params_snapshot=params_snapshot,
    )


def _non_hole_count(result: MetalDetectionResult) -> int:
    return sum(not polygon.is_hole for polygon in result.accepted)


def _mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None or first.shape[:2] != second.shape[:2]:
        return 0.0
    first_active = ensure_binary_mask(first) > 0
    second_active = ensure_binary_mask(second) > 0
    union = int(np.logical_or(first_active, second_active).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(first_active, second_active).sum() / union)


def _prefer_contrast_refined_legacy(
    *,
    legacy: MetalDetectionResult,
    refined: MetalDetectionResult,
    watershed: MetalDetectionResult,
) -> bool:
    """Accept a stricter Otsu mask only when it removes stable low-contrast bridges."""
    legacy_count = _non_hole_count(legacy)
    refined_count = _non_hole_count(refined)
    watershed_count = _non_hole_count(watershed)
    if legacy_count <= 0 or watershed_count < 1.15 * legacy_count:
        return False
    if refined_count < legacy_count or refined_count > 1.05 * legacy_count:
        return False

    watershed_mask = watershed.debug_images.get("metal_filtered_mask")
    legacy_agreement = _mask_iou(
        legacy.debug_images.get("metal_filtered_mask"),
        watershed_mask,
    )
    refined_agreement = _mask_iou(
        refined.debug_images.get("metal_filtered_mask"),
        watershed_mask,
    )
    return refined_agreement > legacy_agreement


def _extended_separator_mask(gray: np.ndarray, config: MetalRecoveryConfig) -> np.ndarray | None:
    watershed = gradient_watershed_config_from_object(config)
    seeds = build_conductor_seeds(gray, watershed)
    if seeds is None or not np.any(seeds.groove_seeds):
        return None
    extension = max(5, 4 * int(watershed.valley_span_px) - 3) | 1
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (extension, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, extension))
    horizontal_lines = cv2.morphologyEx(
        seeds.groove_seeds,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    vertical_lines = cv2.morphologyEx(
        seeds.groove_seeds,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )
    kernel = (
        horizontal_kernel
        if cv2.countNonZero(horizontal_lines) >= cv2.countNonZero(vertical_lines)
        else vertical_kernel
    )
    return cv2.morphologyEx(seeds.groove_seeds, cv2.MORPH_CLOSE, kernel)


def _stable_local_refinement(
    gray: np.ndarray,
    config: MetalRecoveryConfig,
    *,
    source_image: np.ndarray | None,
    reference: MetalDetectionResult,
) -> tuple[MetalDetectionResult | None, float | None, list[int]]:
    """Find a source-filter-stable local mask that still agrees with Auto."""
    reference_count = _non_hole_count(reference)
    step = max(0.0, float(config.auto_source_contrast_step))
    if reference_count < 100 or step <= 0.0:
        return None, None, []
    if gray.size / max(1, reference_count) >= 50_000:
        return None, None, []

    local_config = replace(
        config,
        segmentation_strategy="local_adaptive",
        use_wide_conductor_gradient=False,
    )
    variants: list[tuple[float, MetalDetectionResult, int]] = []
    for step_index in range(6):
        source_contrast = min(
            255.0,
            float(config.min_object_source_contrast) + step_index * step,
        )
        candidate = _detect_metalization_explicit(
            gray,
            replace(local_config, min_object_source_contrast=source_contrast),
            source_image=source_image,
            include_axis_debug=False,
        )
        candidate_count = _non_hole_count(candidate)
        variants.append((source_contrast, candidate, candidate_count))
        if len(variants) < 3:
            continue
        previous = variants[-2]
        if abs(previous[2] - candidate_count) > max(
            1,
            round(0.01 * max(previous[2], candidate_count)),
        ):
            continue

        base_count = variants[0][2]
        required_drop = max(1, ceil(0.01 * max(1, base_count)))
        if base_count - previous[2] < required_drop:
            return None, None, [item[2] for item in variants]
        if abs(previous[2] - reference_count) > 0.10 * max(previous[2], reference_count):
            return None, None, [item[2] for item in variants]
        agreement = _mask_iou(
            previous[1].debug_images.get("metal_filtered_mask"),
            reference.debug_images.get("metal_filtered_mask"),
        )
        moderate_agreement_with_more_objects = (
            agreement >= 0.80
            and previous[2] > reference_count
            and previous[2] <= 1.05 * reference_count
        )
        if agreement < 0.90 and not moderate_agreement_with_more_objects:
            return None, None, [item[2] for item in variants]
        return previous[1], previous[0], [item[2] for item in variants]
    return None, None, [item[2] for item in variants]


def _directionally_bridge_mask(
    mask: np.ndarray,
    source_gray: np.ndarray,
    *,
    bridge_px: int,
    min_source_intensity: float,
) -> np.ndarray:
    """Bridge horizontal gaps only when the missing source pixels remain bright."""
    radius = max(0, int(bridge_px))
    if radius <= 0 or mask.size == 0:
        return mask.copy()
    selected = ensure_uint8(mask).copy()
    _base_count, base_labels = cv2.connectedComponents(selected, connectivity=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 1))
    closed_all = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
    _closed_count, closed_labels = cv2.connectedComponents(closed_all, connectivity=8)
    required_source = max(0.0, float(min_source_intensity))
    for closed_id in np.unique(closed_labels[closed_labels > 0]):
        region = closed_labels == closed_id
        base_ids = np.unique(base_labels[region & (base_labels > 0)])
        if base_ids.size <= 1:
            continue
        added = region & (selected == 0)
        bridge_added = np.zeros_like(added)
        for row in np.flatnonzero(np.any(added, axis=1)):
            occupied_columns = np.flatnonzero(region[row] & (selected[row] > 0))
            if occupied_columns.size < 2:
                continue
            left = int(occupied_columns.min())
            right = int(occupied_columns.max()) + 1
            bridge_added[row, left:right] = added[row, left:right]
        if bridge_added.any() and float(np.median(source_gray[bridge_added])) >= required_source:
            selected[bridge_added] = 255
    return selected


def _refine_local_candidate(
    gray: np.ndarray,
    config: MetalRecoveryConfig,
    *,
    source_image: np.ndarray | None,
    candidate: MetalDetectionResult,
    reference: MetalDetectionResult,
) -> MetalDetectionResult:
    candidate_count = _non_hole_count(candidate)
    if source_image is None or candidate_count <= 0:
        return candidate
    pixels_per_object = gray.size / candidate_count
    if pixels_per_object < 10_000:
        return candidate
    candidate_mask = candidate.debug_images.get("metal_filtered_mask")
    reference_mask = reference.debug_images.get("metal_filtered_mask")
    if candidate_mask is None or reference_mask is None:
        return candidate
    if source_image.ndim == 3:
        source_gray = cv2.cvtColor(ensure_uint8(source_image), cv2.COLOR_BGR2GRAY)
    else:
        source_gray = ensure_uint8(source_image)

    refined_mask = candidate_mask.copy()
    gradient_x = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    x_energy = float(np.mean(np.minimum(gradient_x, 255.0)))
    y_energy = float(np.mean(np.minimum(gradient_y, 255.0)))
    if y_energy >= 1.5 * max(0.1, x_energy):
        refined_mask = _directionally_bridge_mask(
            refined_mask,
            source_gray,
            bridge_px=config.auto_directional_gap_bridge_px,
            min_source_intensity=config.auto_directional_gap_min_source_intensity,
        )

    _candidate_components, candidate_labels = cv2.connectedComponents(
        refined_mask,
        connectivity=8,
    )
    _reference_components, reference_labels = cv2.connectedComponents(
        ensure_uint8(reference_mask),
        connectivity=8,
    )
    for reference_id in np.unique(reference_labels[reference_labels > 0]):
        reference_region = reference_labels == reference_id
        if float(np.mean(candidate_labels[reference_region] > 0)) < 0.10:
            refined_mask[reference_region] = 255

    return _detect_metalization_explicit(
        gray,
        config,
        source_image=source_image,
        mask_override=refined_mask,
        include_axis_debug=False,
    )


def detect_metalization(
    image: np.ndarray,
    config: MetalRecoveryConfig,
    *,
    source_image: np.ndarray | None = None,
) -> MetalDetectionResult:
    """Recognize conductors and preserve topology with an automatic dual-path mode."""
    if _requested_segmentation_strategy(config) != "auto":
        return _detect_metalization_explicit(image, config, source_image=source_image)

    if image.ndim == 3:
        gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2GRAY)
    else:
        gray = ensure_uint8(image)
    if gray.size == 0:
        return MetalDetectionResult(params_snapshot=config.to_snapshot())

    legacy_config = replace(
        config,
        segmentation_strategy="legacy_otsu",
        use_wide_conductor_gradient=False,
    )
    watershed_config = replace(
        config,
        segmentation_strategy="gradient_watershed",
        use_wide_conductor_gradient=True,
    )
    legacy = _detect_metalization_explicit(gray, legacy_config, source_image=source_image)
    watershed = _detect_metalization_explicit(gray, watershed_config, source_image=source_image)
    legacy_count = _non_hole_count(legacy)
    watershed_count = _non_hole_count(watershed)
    presence = analyze_metal_presence(
        gray,
        smoothing_sigma=float(config.watershed_smoothing_sigma),
    )
    coherence = presence.coherent_contrast_fraction

    selected = legacy
    selected_strategy = "legacy_otsu"
    selected_min_contrast = float(config.min_contrast)
    contrast_refined_count: int | None = None
    if not presence.has_metal:
        selected = watershed
        selected_strategy = "gradient_watershed"
    elif watershed_count == 0:
        selected = legacy
        selected_strategy = "legacy_otsu"
    elif (
        coherence < 0.30
        or legacy_count == 0
        or watershed_count <= 0.80 * legacy_count
    ):
        selected = watershed
        selected_strategy = "gradient_watershed"
    else:
        pixels_per_legacy_object = gray.size / max(1, legacy_count)
        counts_are_close = abs(watershed_count - legacy_count) <= 0.05 * max(
            legacy_count,
            watershed_count,
        )
        if pixels_per_legacy_object < 10_000 and counts_are_close:
            selected = watershed
            selected_strategy = "gradient_watershed"
        elif legacy_count <= 0.90 * watershed_count and pixels_per_legacy_object < 10_000:
            separators = _extended_separator_mask(gray, config)
            if separators is not None:
                legacy_mask = legacy.debug_images.get("metal_binary_mask")
                if legacy_mask is not None:
                    hybrid_mask = cv2.bitwise_and(
                        legacy_mask,
                        cv2.bitwise_not(separators),
                    )
                    hybrid = _detect_metalization_explicit(
                        gray,
                        legacy_config,
                        source_image=source_image,
                        mask_override=hybrid_mask,
                    )
                    hybrid_count = _non_hole_count(hybrid)
                    if legacy_count < hybrid_count <= round(1.05 * watershed_count):
                        selected = hybrid
                        selected_strategy = "legacy_otsu_extended_separators"

    if selected is legacy and watershed_count >= 1.15 * legacy_count:
        refined_min_contrast = min(
            255.0,
            float(config.min_contrast) + max(0.0, float(config.auto_contrast_step)),
        )
        if refined_min_contrast > float(config.min_contrast):
            refined_config = replace(legacy_config, min_contrast=refined_min_contrast)
            refined = _detect_metalization_explicit(
                gray,
                refined_config,
                source_image=source_image,
            )
            contrast_refined_count = _non_hole_count(refined)
            if _prefer_contrast_refined_legacy(
                legacy=legacy,
                refined=refined,
                watershed=watershed,
            ):
                selected = refined
                selected_strategy = "legacy_otsu_contrast_refined"
                selected_min_contrast = refined_min_contrast

    local_refined, local_source_contrast, local_counts = _stable_local_refinement(
        gray,
        config,
        source_image=source_image,
        reference=selected,
    )
    if local_refined is not None:
        assert local_source_contrast is not None
        local_config = replace(
            config,
            segmentation_strategy="local_adaptive",
            use_wide_conductor_gradient=False,
            min_object_source_contrast=float(local_source_contrast),
        )
        local_refined = _refine_local_candidate(
            gray,
            local_config,
            source_image=source_image,
            candidate=local_refined,
            reference=selected,
        )
        selected = local_refined
        selected_strategy = "local_adaptive_source_refined"
        selected.debug_images.update(axis_gradient_debug_images(gray))

    selected.params_snapshot = {
        **config.to_snapshot(),
        "auto_selected_strategy": selected_strategy,
        "auto_selected_min_contrast": selected_min_contrast,
        "auto_legacy_objects": legacy_count,
        "auto_watershed_objects": watershed_count,
        "auto_contrast_refined_objects": contrast_refined_count,
        "auto_local_source_contrast": local_source_contrast,
        "auto_local_objects": local_counts,
    }
    return selected
