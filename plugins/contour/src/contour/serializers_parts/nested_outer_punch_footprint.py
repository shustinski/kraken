"""serializers: nested outer punch footprint."""
from __future__ import annotations
from contour.domain import PolygonData

def _family_geometry(outer: PolygonData, holes: list[PolygonData]):
    from shapely import make_valid
    from shapely.geometry import Polygon as ShapelyPolygon

    shell = [(float(x_coord), float(y_coord)) for x_coord, y_coord in outer.points]
    interiors = [
        [(float(x_coord), float(y_coord)) for x_coord, y_coord in hole.points]
        for hole in holes
        if len(hole.points) >= 3
    ]
    if len(shell) < 3:
        return ShapelyPolygon()
    return make_valid(ShapelyPolygon(shell, interiors))

def _nested_outer_punch_footprint(small_solid, large_geom):
    """Footprint to subtract from a covering parent for a nested outer.

    Concave outer notches (e.g. a U-cut at the top of a pad island) sit outside the
    island solid but still belong to the nested window. Punching only the solid leaves
    parent metal in those pockets and hides the inner contour. Expand to the convex
    hull when those pockets lie under the covering parent.
    """

    from shapely import make_valid

    hull = make_valid(small_solid.convex_hull)
    if hull.is_empty or float(hull.area) <= float(small_solid.area) + 1.0:
        return small_solid
    extra = make_valid(hull.difference(small_solid))
    if extra.is_empty or float(extra.area) <= 1.0:
        return small_solid
    covered_extra = large_geom.intersection(extra)
    if covered_extra.is_empty or float(covered_extra.area) < 0.5 * float(extra.area):
        return small_solid
    return hull
