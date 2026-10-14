"""serializers: polygons shapely geometry."""
from __future__ import annotations
from contour.domain import PolygonData, compute_polygon_metrics

def _polygons_from_shapely_geometry(geometry, *, start_id: int) -> list[PolygonData]:
    from shapely import make_valid
    from shapely.geometry import GeometryCollection, MultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon

    geom = make_valid(geometry)
    if geom.is_empty:
        return []

    def _collect_polygons(part) -> list:
        if isinstance(part, ShapelyPolygon):
            return [part] if not part.is_empty else []
        if isinstance(part, MultiPolygon):
            return [child for child in part.geoms if isinstance(child, ShapelyPolygon) and not child.is_empty]
        if isinstance(part, GeometryCollection):
            collected: list = []
            for child in part.geoms:
                collected.extend(_collect_polygons(child))
            return collected
        return []

    result: list[PolygonData] = []
    next_id = start_id
    for poly in _collect_polygons(geom):
        if float(poly.area) <= 1.0:
            continue
        exterior = [(float(x_coord), float(y_coord)) for x_coord, y_coord in poly.exterior.coords[:-1]]
        if len(exterior) < 3:
            continue
        area, perimeter, bbox = compute_polygon_metrics(exterior)
        parent_id = next_id
        result.append(
            PolygonData(
                id=parent_id,
                points=exterior,
                is_hole=False,
                parent_id=None,
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )
        next_id += 1
        for interior in poly.interiors:
            hole_points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in interior.coords[:-1]]
            if len(hole_points) < 3:
                continue
            area, perimeter, bbox = compute_polygon_metrics(hole_points)
            result.append(
                PolygonData(
                    id=next_id,
                    points=hole_points,
                    is_hole=True,
                    parent_id=parent_id,
                    area=area,
                    perimeter=perimeter,
                    bbox=bbox,
                )
            )
            next_id += 1
    return result
