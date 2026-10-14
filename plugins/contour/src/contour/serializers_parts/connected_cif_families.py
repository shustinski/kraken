"""serializers: connected cif families."""
from __future__ import annotations
from contour.domain import PolygonData

from contour.serializers_parts.polygons_shapely_geometry import _polygons_from_shapely_geometry

def _connected_cif_families(
    outer: PolygonData,
    holes: list[PolygonData],
) -> list[tuple[PolygonData, list[PolygonData]]]:
    """Split disconnected material like KLayout's resolve-holes writer stage."""

    from shapely import make_valid, unary_union
    from shapely.geometry import Polygon as ShapelyPolygon

    outer_geometry = make_valid(ShapelyPolygon(outer.points))
    hole_geometries = [
        make_valid(ShapelyPolygon(hole.points))
        for hole in holes
        if len(hole.points) >= 3
    ]
    material = make_valid(
        outer_geometry.difference(unary_union(hole_geometries))
        if hole_geometries
        else outer_geometry
    )
    pieces = _polygons_from_shapely_geometry(material, start_id=1)
    piece_outers = [piece for piece in pieces if not piece.is_hole]
    if len(piece_outers) <= 1:
        return [(outer, holes)]

    result: list[tuple[PolygonData, list[PolygonData]]] = []
    for piece_outer in piece_outers:
        piece_outer.category = str(outer.category)
        piece_outer.shape_hint = str(outer.shape_hint)
        piece_holes = [piece for piece in pieces if piece.is_hole and piece.parent_id == piece_outer.id]
        for piece_hole in piece_holes:
            piece_hole.category = str(outer.category)
            piece_hole.shape_hint = str(outer.shape_hint)
        result.append((piece_outer, piece_holes))
    return result
