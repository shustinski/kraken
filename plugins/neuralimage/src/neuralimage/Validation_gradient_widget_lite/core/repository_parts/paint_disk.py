"""repository: paint disk."""
from __future__ import annotations
import math
import numpy as np

def _paint_disk(target: np.ndarray, center_x: float, center_y: float, radius: float, value: float) -> None:
    if target.ndim != 2 or target.size == 0:
        return
    effective_radius = max(1.0, float(radius))
    x = int(round(float(center_x)))
    y = int(round(float(center_y)))
    y0 = max(0, int(math.floor(y - effective_radius)))
    y1 = min(target.shape[0], int(math.ceil(y + effective_radius + 1.0)))
    x0 = max(0, int(math.floor(x - effective_radius)))
    x1 = min(target.shape[1], int(math.ceil(x + effective_radius + 1.0)))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2, dtype=np.float64)
    disk = distance <= effective_radius
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], np.asarray(disk, dtype=np.float32) * float(value))

def _point_map_from_view(prediction_view: object, shape: tuple[int, int] | None = None, *, scale: float = 1.0) -> np.ndarray:
    base_shape = tuple(int(v) for v in (shape or getattr(prediction_view, 'pred_gray').shape))
    result = np.zeros(base_shape, dtype=np.float32)
    for point in tuple(getattr(prediction_view, 'points', ())):
        radius = max(1.0, float(getattr(point, 'radius', 0.0)))
        score = float(getattr(point, 'score', 1.0) or 1.0)
        _paint_disk(result, float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), radius + 1.0, float(scale) * float(max(0.25, min(1.0, score))))
    return np.clip(result, 0.0, 1.0)
