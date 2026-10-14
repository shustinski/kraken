"""detector: polygon interior median."""
from __future__ import annotations
import cv2
import numpy as np
from contour.domain import PolygonData

def _renumber_polygons_preserving_parents(polygons: list[PolygonData]) -> None:
    if not polygons:
        return
    old_to_new: dict[int, int] = {}
    for new_id, poly in enumerate(polygons, start=1):
        old_to_new[int(poly.id)] = new_id
    for poly in polygons:
        poly.id = old_to_new[int(poly.id)]
    for poly in polygons:
        if poly.parent_id is not None:
            poly.parent_id = old_to_new.get(int(poly.parent_id), poly.parent_id)

def _polygon_interior_median(
    gray: np.ndarray,
    polygon: PolygonData,
    *,
    excluded: list[PolygonData] | None = None,
) -> float | None:
    points = np.asarray(polygon.points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    height, width = gray.shape[:2]
    x_min = max(0, int(np.floor(points[:, 0].min())))
    y_min = max(0, int(np.floor(points[:, 1].min())))
    x_max = min(width - 1, int(np.ceil(points[:, 0].max())))
    y_max = min(height - 1, int(np.ceil(points[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        return None

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)

    def _local_points(item: PolygonData) -> np.ndarray:
        item_points = np.asarray(item.points, dtype=np.float32).reshape(-1, 2)
        item_points[:, 0] -= x_min
        item_points[:, 1] -= y_min
        return np.rint(item_points).astype(np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(mask, [_local_points(polygon)], 255)
    for item in excluded or []:
        if len(item.points) >= 3:
            cv2.fillPoly(mask, [_local_points(item)], 0)
    values = gray[y_min : y_max + 1, x_min : x_max + 1][mask > 0]
    if values.size == 0:
        return None
    return float(np.median(values))
