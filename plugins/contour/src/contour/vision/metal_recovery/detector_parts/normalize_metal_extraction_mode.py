"""detector: normalize metal extraction mode."""
from __future__ import annotations
from typing import Any
from contour.vision.metal_recovery.segmentation import (
    normalize_metal_segmentation_strategy,
)

def _normalize_metal_extraction_mode(value: Any) -> str:
    """Compatibility bridge to the canonical segmentation normalizer."""
    return normalize_metal_segmentation_strategy(value)
