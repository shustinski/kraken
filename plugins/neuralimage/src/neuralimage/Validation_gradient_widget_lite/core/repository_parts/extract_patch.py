"""repository: extract patch."""
from __future__ import annotations
import numpy as np

def _extract_patch(image: np.ndarray, center_x: float, center_y: float, radius: int = 2) -> np.ndarray:
    height, width = image.shape
    x = int(round(center_x))
    y = int(round(center_y))
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    return image[y0:y1, x0:x1]
