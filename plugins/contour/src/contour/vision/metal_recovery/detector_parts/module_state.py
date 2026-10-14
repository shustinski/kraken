"""detector: module state."""
from __future__ import annotations

_CONTOUR_CACHE: dict[str, _ContourStageCache] = {}

_CONTOUR_CACHE_MAX_ITEMS = 8
