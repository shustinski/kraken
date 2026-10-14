"""_core: render polygon mask polygons."""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from contour.domain import PolygonData
from contour.utils import ensure_binary_mask

def _render_polygon_mask_from_polygons(source: Any, polygons: list[PolygonData]) -> np.ndarray:
    shape = np.asarray(source).shape[:2]
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon.points) < 3:
            continue
        pts = np.array([[round(x), round(y)] for x, y in polygon.points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return ensure_binary_mask(mask)
