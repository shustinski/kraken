"""serializers: punch nested outer covers."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.serializers_parts.nested_outer_punch_footprint import (
    _family_geometry,
    _nested_outer_punch_footprint,
)

from contour.serializers_parts.polygons_shapely_geometry import _polygons_from_shapely_geometry

def _punch_nested_outer_covers(polygons: list[PolygonData]) -> list[PolygonData]:
    """Subtract smaller outers from larger overlapping fills (opt-in topology rewrite).

    Not applied on normal CIF load: overlapping frame+island metal is valid CIF and
    matches KLayout OR paint. When enabled (see ``_nested_outer_punch_on_load_enabled``),
    each nested outer's window footprint (solid, expanded to convex hull when concave
    pockets sit under the parent) is carved from covering parents so island holes open
    onto empty space instead of parent metal.
    """

    from contour.serializers_parts.cif_box_polygon import _clone_polygon_with_id
    from shapely import make_valid
    from shapely.geometry import Point
    from shapely.geometry import Polygon as ShapelyPolygon

    outers = [polygon for polygon in polygons if not polygon.is_hole]
    if len(outers) < 2:
        return polygons
    holes_by_parent: dict[int, list[PolygonData]] = {}
    for polygon in polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)

    geoms = {
        int(outer.id): _family_geometry(outer, holes_by_parent.get(int(outer.id), []))
        for outer in outers
    }
    originals = {int(outer.id): outer for outer in outers}
    changed_ids: set[int] = set()

    for large_id in sorted(originals, key=lambda item_id: -float(originals[item_id].area)):
        large_geom = geoms.get(large_id)
        if large_geom is None or large_geom.is_empty:
            continue
        for small_id, small_outer in originals.items():
            if small_id == large_id:
                continue
            if float(small_outer.area) >= float(originals[large_id].area) * 0.5:
                continue
            if len(small_outer.points) < 3:
                continue
            center = Point(
                float(small_outer.bbox[0]) + float(small_outer.bbox[2]) * 0.5,
                float(small_outer.bbox[1]) + float(small_outer.bbox[3]) * 0.5,
            )
            if not large_geom.contains(center):
                continue
            small_solid = make_valid(
                ShapelyPolygon([(float(x_coord), float(y_coord)) for x_coord, y_coord in small_outer.points])
            )
            if small_solid.is_empty or float(small_solid.area) <= 1.0:
                continue
            overlap = large_geom.intersection(small_solid)
            if overlap.is_empty or float(overlap.area) < 0.5 * float(small_solid.area):
                continue
            punch_footprint = _nested_outer_punch_footprint(small_solid, large_geom)
            punched = make_valid(large_geom.difference(punch_footprint))
            if punched.is_empty:
                continue
            if abs(float(punched.area) - float(large_geom.area)) < 1.0:
                continue
            geoms[large_id] = punched
            large_geom = punched
            changed_ids.add(large_id)

    if not changed_ids:
        return polygons

    rebuilt: list[PolygonData] = []
    next_id = 1
    for outer in sorted(outers, key=lambda item: int(item.id)):
        outer_id = int(outer.id)
        family_holes = holes_by_parent.get(outer_id, [])
        if outer_id not in changed_ids:
            clone = _clone_polygon_with_id(outer, next_id)
            parent_id = next_id
            rebuilt.append(clone)
            next_id += 1
            for hole in family_holes:
                hole_clone = _clone_polygon_with_id(hole, next_id)
                hole_clone.is_hole = True
                hole_clone.parent_id = parent_id
                rebuilt.append(hole_clone)
                next_id += 1
            continue
        pieces = _polygons_from_shapely_geometry(geoms[outer_id], start_id=next_id)
        if not pieces:
            clone = _clone_polygon_with_id(outer, next_id)
            rebuilt.append(clone)
            next_id += 1
            continue
        for piece in pieces:
            piece.category = str(outer.category)
            piece.shape_hint = str(outer.shape_hint)
            rebuilt.append(piece)
            next_id = max(next_id, int(piece.id) + 1)
    return rebuilt
