"""vector_geometry_postprocess: rebuild family filled mask."""
from __future__ import annotations
import numpy as np
from contour.domain import PolygonData

from contour.application.vector_geometry_postprocess_parts.collapse_redundant_vertices_polygon import (
    _mask_helpers,
    _refresh_metrics,
)

from contour.application.vector_geometry_postprocess_parts.root_id_map import _stamp_visual_metadata

def _rebuild_family_from_filled_mask(family: list[PolygonData]) -> list[PolygonData]:
    _bbox_from_points, _polygons_from_mask, _render_polygon_collection_on_mask, _union_bbox = _mask_helpers()
    del _union_bbox
    points = [(x_coord, y_coord) for polygon in family for x_coord, y_coord in polygon.points]
    if not points:
        return []
    origin_x, origin_y, width, height = _bbox_from_points(points, padding=4)
    mask = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
    _render_polygon_collection_on_mask(mask, [polygon.clone() for polygon in family], (origin_x, origin_y))
    extracted = _polygons_from_mask(mask, (origin_x, origin_y))
    _stamp_visual_metadata(family, extracted)
    for polygon in extracted:
        _refresh_metrics(polygon)
    return extracted

def _polygon_ring_needs_dissolve(polygon: PolygonData) -> bool:
    return polygon.description_is_invalid()

def _polygon_ring_has_keyhole_bridge(polygon: PolygonData) -> bool:
    return (not polygon.is_hole) and polygon.description_invalid_reason() == "repeated_vertex"

def _polygon_ring_needs_explicit_topology_repair(polygon: PolygonData) -> bool:
    """True for bowties or CIF keyholes when the user explicitly requests repair."""

    return _polygon_ring_needs_dissolve(polygon) or _polygon_ring_has_keyhole_bridge(polygon)
