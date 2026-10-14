"""vector_geometry_postprocess: repair family by keyhole."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics

from contour.application.vector_geometry_postprocess_parts.rebuild_family_filled_mask import _polygon_ring_needs_explicit_topology_repair

def _repair_family_by_keyhole_split(family: list[PolygonData]) -> list[PolygonData] | None:
    """Split CIF keyhole polylines into outer+holes without reshaping via make_valid.

    Shapely dissolve changes the filled outline; keyhole extraction keeps the authored
    rings and only cuts the bridge that made the description invalid.
    """

    from contour.serializers import _split_linked_polygon_rings

    if not any(
        (not polygon.is_hole) and polygon.description_invalid_reason() == "repeated_vertex"
        for polygon in family
    ):
        return None

    current = [polygon.clone() for polygon in family]
    changed = False
    for _pass in range(8):
        next_id = max((polygon.id for polygon in current), default=0) + 1
        rebuilt: list[PolygonData] = []
        progress = False
        for polygon in current:
            if polygon.is_hole or polygon.description_invalid_reason() != "repeated_vertex":
                rebuilt.append(polygon)
                continue
            outer_points, hole_rings = _split_linked_polygon_rings(polygon.points)
            if not hole_rings:
                rebuilt.append(polygon)
                continue
            progress = True
            changed = True
            area, perimeter, bbox = compute_polygon_metrics(outer_points)
            outer = polygon.clone()
            outer.points = list(outer_points)
            outer.area = float(area)
            outer.perimeter = float(perimeter)
            outer.bbox = bbox
            rebuilt.append(outer)
            for hole_points in hole_rings:
                area, perimeter, bbox = compute_polygon_metrics(hole_points)
                rebuilt.append(
                    PolygonData(
                        id=next_id,
                        points=list(hole_points),
                        is_hole=True,
                        parent_id=outer.id,
                        category=str(polygon.category),
                        shape_hint=str(polygon.shape_hint),
                        area=float(area),
                        perimeter=float(perimeter),
                        bbox=bbox,
                    )
                )
                next_id += 1
        current = rebuilt
        if not progress:
            break

    if not changed:
        return None
    if any(_polygon_ring_needs_explicit_topology_repair(polygon) for polygon in current):
        return None
    return current
