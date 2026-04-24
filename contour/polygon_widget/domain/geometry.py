from __future__ import annotations

from collections.abc import Iterable
from math import ceil, floor, hypot

from .polygon import Point


def compute_polygon_metrics(points: Iterable[Point]) -> tuple[float, float, tuple[int, int, int, int]]:
    vertices = [(float(x_coord), float(y_coord)) for x_coord, y_coord in points]
    if not vertices:
        return 0.0, 0.0, (0, 0, 0, 0)
    if len(vertices) == 1:
        x_coord, y_coord = vertices[0]
        return 0.0, 0.0, (floor(x_coord), floor(y_coord), 1, 1)
    if len(vertices) == 2:
        (x0, y0), (x1, y1) = vertices
        perimeter = hypot(x1 - x0, y1 - y0) * 2.0
        x_min = floor(min(x0, x1))
        y_min = floor(min(y0, y1))
        x_max = ceil(max(x0, x1))
        y_max = ceil(max(y0, y1))
        return 0.0, perimeter, (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

    area_sum = 0.0
    perimeter = 0.0
    wrapped = vertices[1:] + vertices[:1]
    for (x0, y0), (x1, y1) in zip(vertices, wrapped, strict=False):
        area_sum += x0 * y1 - x1 * y0
        perimeter += hypot(x1 - x0, y1 - y0)

    x_values = [point[0] for point in vertices]
    y_values = [point[1] for point in vertices]
    x_min = floor(min(x_values))
    y_min = floor(min(y_values))
    x_max = ceil(max(x_values))
    y_max = ceil(max(y_values))
    return abs(area_sum) / 2.0, perimeter, (x_min, y_min, max(1, x_max - x_min + 1), max(1, y_max - y_min + 1))
