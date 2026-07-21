"""Conductor recognition: segmentation mask → findContours + post-filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ...application.processing import ContourExtractionSettings
from ...contour_extractor import extract_polygons
from ...domain import PolygonData
from ...utils import ensure_binary_mask, ensure_uint8

from .pipeline_stages import build_metal_segmentation_mask_staged, image_signature
from .segmentation import MetalSegmentationConfig, normalize_metal_segmentation_strategy


def _normalize_metal_extraction_mode(value: Any) -> str:
    """Compatibility bridge to the canonical segmentation normalizer."""
    return normalize_metal_segmentation_strategy(value)


@dataclass(slots=True)
class MetalRecoveryConfig:
    """Segmentation + contour extraction parameters exposed in the UI."""

    contrast_bias: float = 0.0
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

    def to_snapshot(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in MetalRecoveryConfig.__dataclass_fields__.values()
        }


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
    wide_gradient_overlays: dict[str, list[PolygonData]] = field(default_factory=dict)


@dataclass(slots=True)
class _ContourStageCache:
    mask_sig: str = ""
    config_sig: tuple[Any, ...] = ()
    polygons: list[PolygonData] = field(default_factory=list)


_CONTOUR_CACHE: dict[str, _ContourStageCache] = {}
_CONTOUR_CACHE_MAX_ITEMS = 8


def clear_metal_contour_cache() -> None:
    _CONTOUR_CACHE.clear()


def _segmentation_config_from_recovery(config: MetalRecoveryConfig) -> MetalSegmentationConfig:
    return MetalSegmentationConfig(
        contrast_bias=float(config.contrast_bias),
        gap_bridge_px=max(0, int(config.gap_bridge_px)),
        speckle_removal_px=max(0, int(config.speckle_removal_px)),
        min_component_area=max(0, int(config.min_component_area or config.min_area)),
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
    if config.max_width_px is not None and config.max_width_px > 0.0:
        if _trace_width_px(polygon) > float(config.max_width_px):
            return False
    return True


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


def detect_metalization(image: np.ndarray, config: MetalRecoveryConfig) -> MetalDetectionResult:
    """Recognition: Otsu mask → findContours (epsilon) → geometric filters."""
    if image.ndim == 3:
        gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2GRAY)
    else:
        gray = ensure_uint8(image)
    if gray.size == 0:
        return MetalDetectionResult(params_snapshot=config.to_snapshot())

    mask, pre_dbg = build_metal_extraction_mask(gray, config)
    raise_if_preview_cancelled()

    h, w = mask.shape[:2]
    polygons = _extract_polygons_cached(mask, config)

    accepted: list[PolygonData] = []
    border: list[MetalPolygonRecord] = []
    accepted_mask = np.zeros_like(mask)
    next_id = 1
    border_mode = str(config.border_mode or "mark").strip().lower()

    for polygon in polygons:
        if not _passes_trace_geometry(polygon, config):
            continue

        poly = polygon.clone()
        if polygon.is_hole:
            poly.id = next_id
            next_id += 1
            poly.category = "conductor"
            accepted.append(poly)
            continue

        touches_border = _border_touch_points(poly.points, width=w, height=h)
        if border_mode == "ignore" and touches_border:
            continue

        poly.id = next_id
        next_id += 1
        poly.category = "conductor"
        if border_mode == "mark" and touches_border:
            border.append(
                MetalPolygonRecord(
                    polygon=poly.clone(),
                    area=float(poly.area),
                    perimeter=float(poly.perimeter),
                    border_touch=True,
                )
            )
            poly.category = "metal_border"

        accepted.append(poly)
        pts = np.array(poly.points, dtype=np.int32).reshape(-1, 1, 2)
        if pts.shape[0] >= 3:
            cv2.fillPoly(accepted_mask, [pts], 255)

    _renumber_polygons_preserving_parents(accepted)

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
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if raw_contours:
        cv2.drawContours(vis, raw_contours, -1, (0, 255, 0), 1)
    dbg["metal_contours_raw"] = vis
    dbg["metal_width_check"] = cv2.cvtColor(accepted_mask, cv2.COLOR_GRAY2BGR)

    return MetalDetectionResult(
        accepted=accepted,
        rejected=[],
        suspicious=[],
        border=border,
        debug_images=dbg,
        params_snapshot=config.to_snapshot(),
    )
