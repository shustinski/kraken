"""detector: rasterize accepted conductors."""
from __future__ import annotations
import cv2
import numpy as np
from contour.domain import PolygonData

def _rasterize_accepted_conductors(
    polygons: list[PolygonData],
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a filled mask while preserving holes and nested conductors."""
    holes_by_parent: dict[int, list[PolygonData]] = {}
    conductors: list[tuple[float, PolygonData]] = []
    for polygon in polygons:
        if polygon.is_hole:
            if polygon.parent_id is not None:
                holes_by_parent.setdefault(int(polygon.parent_id), []).append(polygon)
            continue
        conductors.append((abs(float(polygon.area)), polygon))

    mask = np.zeros(shape, dtype=np.uint8)
    for _area, conductor in sorted(conductors, key=lambda item: item[0], reverse=True):
        points = np.asarray(conductor.points, dtype=np.int32).reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        x_min = max(0, int(points[:, 0].min()))
        x_max = min(shape[1] - 1, int(points[:, 0].max()))
        y_min = max(0, int(points[:, 1].min()))
        y_max = min(shape[0] - 1, int(points[:, 1].max()))
        if x_max < x_min or y_max < y_min:
            continue
        region = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        local_points = points.copy()
        local_points[:, 0] -= x_min
        local_points[:, 1] -= y_min
        cv2.fillPoly(region, [local_points.reshape(-1, 1, 2)], 255)
        for hole in holes_by_parent.get(int(conductor.id), ()):
            hole_points = np.asarray(hole.points, dtype=np.int32).reshape(-1, 2)
            if hole_points.shape[0] < 3:
                continue
            local_hole_points = hole_points.copy()
            local_hole_points[:, 0] -= x_min
            local_hole_points[:, 1] -= y_min
            cv2.fillPoly(region, [local_hole_points.reshape(-1, 1, 2)], 0)
        target = mask[y_min : y_max + 1, x_min : x_max + 1]
        target[region > 0] = 255
    return mask
