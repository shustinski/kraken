"""serializers: normalize cif polygon families."""
from __future__ import annotations

from contour.serializers_parts.cif_paint_mask_ring import _ring_coords_from_shapely

def _normalize_cif_polygon_families(
    points: list[tuple[float, float]],
) -> list[tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]] | None:
    """Normalize a CIF ``P`` ring to outer+holes via make_valid (KLayout-like fill).

    Heuristic keyhole splitting can invent extra holes and change filled area. Shapely
    ``make_valid`` matches the geometric fill of self-touching CIF polygons that layout
    viewers normalize when rendering.
    """

    from shapely import make_valid
    from shapely.geometry import GeometryCollection, MultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon

    if len(points) < 3:
        return None
    shell = [(float(x_coord), float(y_coord)) for x_coord, y_coord in points]
    geom = make_valid(ShapelyPolygon(shell))
    if geom.is_empty:
        return None

    polygons: list = []

    def _collect(part) -> None:
        if part.is_empty:
            return
        if isinstance(part, ShapelyPolygon):
            polygons.append(part)
        elif isinstance(part, MultiPolygon):
            polygons.extend(child for child in part.geoms if isinstance(child, ShapelyPolygon))
        elif isinstance(part, GeometryCollection):
            for child in part.geoms:
                _collect(child)

    _collect(geom)
    if not polygons:
        return None

    families: list[tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]] = []
    for poly in polygons:
        if float(poly.area) <= 1.0:
            continue
        outer = _ring_coords_from_shapely(poly.exterior.coords)
        if len(outer) < 3:
            continue
        holes: list[list[tuple[float, float]]] = []
        for interior in poly.interiors:
            hole = _ring_coords_from_shapely(interior.coords)
            if len(hole) >= 3:
                holes.append(hole)
        families.append((outer, holes))
    return families or None
