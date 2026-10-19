"""detector: module state."""
from __future__ import annotations
from typing import TYPE_CHECKING

_CONTOUR_CACHE: dict[str, _ContourStageCache] = {}

_CONTOUR_CACHE_MAX_ITEMS = 8

if TYPE_CHECKING:
    from contour.vision.metal_recovery.detector_parts.contour_extraction_settings_metal import _ContourStageCache
