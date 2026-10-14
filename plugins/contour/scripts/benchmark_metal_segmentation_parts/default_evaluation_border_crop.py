"""benchmark_metal_segmentation: default evaluation border crop."""
from __future__ import annotations

from benchmark_metal_segmentation_parts.module_state import EVALUATION_BORDER_CROP_PX

def _default_evaluation_border_crop_px(suite: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    return EVALUATION_BORDER_CROP_PX if suite == "real" else 0
