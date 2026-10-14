"""repository: compute comparison."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    ComparisonMode,
)

def compute_comparison(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> tuple[np.ndarray, float]:
    if mode == ComparisonMode.GRAYSCALE_DIFF:
        first_gray = np.asarray(first, dtype=np.float32)
        second_gray = np.asarray(second, dtype=np.float32)
        heatmap = np.abs(first_gray - second_gray)
        if heatmap.size == 0:
            return heatmap.astype(np.float32), 0.0
        scale = 255.0 if max(float(np.max(first_gray)), float(np.max(second_gray))) > 1.0 else 1.0
        if scale > 1.0:
            heatmap = heatmap / scale
        return heatmap.astype(np.float32, copy=False), float(np.mean(heatmap, dtype=np.float64))
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if mode == ComparisonMode.OVERLAY_ONLY:
        return np.zeros_like(first_bool, dtype=np.float32), 0.0
    elif mode in {ComparisonMode.XOR, ComparisonMode.DISAGREEMENT}:
        mask = np.logical_xor(first_bool, second_bool)
    elif mode == ComparisonMode.FIRST_MINUS_SECOND:
        mask = np.logical_and(first_bool, np.logical_not(second_bool))
    else:
        mask = np.logical_and(np.logical_not(first_bool), second_bool)
    return mask.astype(np.float32), float(np.mean(mask, dtype=np.float64))
