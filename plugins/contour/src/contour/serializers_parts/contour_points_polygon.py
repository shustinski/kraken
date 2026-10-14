"""serializers: contour points polygon."""
from __future__ import annotations
import numpy as np
from contour.domain import PolygonData, compute_polygon_metrics

def _contour_points_to_polygon(
    contour: np.ndarray,
    *,
    left: int,
    top: int,
    polygon_id: int,
    is_hole: bool,
    parent_id: int | None,
) -> PolygonData:
    points = [(float(point[0][0] + left), float(point[0][1] + top)) for point in contour]
    area, perimeter, bbox = compute_polygon_metrics(points)
    return PolygonData(
        id=polygon_id,
        points=points,
        is_hole=is_hole,
        parent_id=parent_id,
        category="conductor",
        shape_hint="polygon",
        area=area,
        perimeter=perimeter,
        bbox=bbox,
    )
