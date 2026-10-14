"""detector: filter holes by source."""
from __future__ import annotations
import cv2
import numpy as np
from contour.domain import PolygonData
from contour.utils import ensure_uint8

from contour.vision.metal_recovery.detector_parts.polygon_interior_median import _polygon_interior_median

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
