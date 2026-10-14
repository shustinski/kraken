"""bright_tophat_dog: center distance index."""
from __future__ import annotations
import numpy as np

def _clip01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)

class _CenterDistanceIndex:
    """Bucketing for O(1) amortized "any center within min_dist" checks on a 2D grid."""

    __slots__ = ("_buckets", "_cell", "_min_dist")

    def __init__(self, min_dist: float) -> None:
        self._min_dist = float(max(0.0, min_dist))
        self._cell = max(1.0, self._min_dist)
        self._buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def is_close(self, cx: float, cy: float) -> bool:
        if self._min_dist <= 0.0:
            return False
        c = self._cell
        ix, iy = int(cx // c), int(cy // c)
        md = self._min_dist
        for dxi in (-1, 0, 1):
            for dyi in (-1, 0, 1):
                for ox, oy in self._buckets.get((ix + dxi, iy + dyi), ()):
                    if float(np.hypot(cx - ox, cy - oy)) < md:
                        return True
        return False

    def add(self, cx: float, cy: float) -> None:
        c = self._cell
        self._buckets.setdefault((int(cx // c), int(cy // c)), []).append((cx, cy))
