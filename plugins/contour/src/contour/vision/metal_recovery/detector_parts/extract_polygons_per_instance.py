"""detector: extract polygons per instance."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from contour.contour_extractor import extract_polygons
from contour.domain import PolygonData
from contour.utils import ensure_uint8
from contour.vision.metal_recovery.pipeline_stages import image_signature

from contour.vision.metal_recovery.detector_parts.module_state import (
    _CONTOUR_CACHE,
    _CONTOUR_CACHE_MAX_ITEMS,
)

from contour.vision.metal_recovery.detector_parts.contour_extraction_settings_metal import (
    _ContourStageCache,
    _contour_config_signature,
    contour_extraction_settings_from_metal,
)

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

if TYPE_CHECKING:
    from contour.vision.metal_recovery.detector_parts.metal_recovery_config import MetalRecoveryConfig
