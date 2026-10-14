"""detector: detect metalization explicit."""
from __future__ import annotations
import cv2
import numpy as np
from ....application.preview_cancellation import raise_if_preview_cancelled
from contour.domain import PolygonData
from contour.utils import ensure_binary_mask, ensure_uint8
from contour.vision.metal_recovery.pipeline_stages import axis_gradient_debug_images, build_metal_segmentation_mask_staged
from contour.vision.metal_recovery.strategy_registry import IMPLEMENTED_NEW_STRATEGIES

from contour.vision.metal_recovery.detector_parts.contour_extraction_settings_metal import (
    MetalDetectionResult,
    MetalPolygonRecord,
    _border_touch_points,
    _passes_trace_geometry,
    _requested_segmentation_strategy,
    _segmentation_config_from_recovery,
)

from contour.vision.metal_recovery.detector_parts.extract_polygons_per_instance import (
    _extract_polygons_cached,
    _extract_polygons_per_instance,
)

from contour.vision.metal_recovery.detector_parts.filter_conductors_by_source import _filter_conductors_by_source_contrast

from contour.vision.metal_recovery.detector_parts.filter_holes_by_source import _filter_holes_by_source_contrast

from contour.vision.metal_recovery.detector_parts.polygon_interior_median import _renumber_polygons_preserving_parents

from contour.vision.metal_recovery.detector_parts.rasterize_accepted_conductors import _rasterize_accepted_conductors

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
    from contour.vision.metal_recovery.structural_watershed import INSTANCE_IDENTITY_VARIANTS

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
