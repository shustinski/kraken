"""repository: load frame layers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    ComparisonMode,
    FrameRecord,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mean_filter_3x3 import (
    load_grayscale_image,
    resize_grayscale_image,
)

def compute_comparison_score(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> float:
    """Legacy-lite helper that returns only scalar score for a comparison mode."""

    if mode == ComparisonMode.GRAYSCALE_DIFF:
        first_gray = np.asarray(first, dtype=np.float32)
        second_gray = np.asarray(second, dtype=np.float32)
        if first_gray.size == 0 or second_gray.size == 0:
            return 0.0
        diff = np.abs(first_gray - second_gray)
        scale = 255.0 if max(float(np.max(first_gray)), float(np.max(second_gray))) > 1.0 else 1.0
        return float(np.mean(diff, dtype=np.float64) / scale)
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if first_bool.size == 0 or second_bool.size == 0 or mode == ComparisonMode.OVERLAY_ONLY:
        return 0.0
    if mode in {ComparisonMode.XOR, ComparisonMode.DISAGREEMENT}:
        return float(np.mean(np.not_equal(first_bool, second_bool), dtype=np.float64))
    if mode == ComparisonMode.FIRST_MINUS_SECOND:
        return float(np.mean(np.logical_and(first_bool, np.logical_not(second_bool)), dtype=np.float64))
    return float(np.mean(np.logical_and(np.logical_not(first_bool), second_bool), dtype=np.float64))

def load_frame_layers(record: FrameRecord) -> dict[str, object]:
    """Legacy-lite helper returning first/second/base grayscale and binary layers."""

    first_path = str(record.first_path or "")
    second_path = str(record.second_path or "")
    if not first_path and record.model_mask_paths:
        first_path = str(next(iter(record.model_mask_paths.values()), ""))
    if not second_path and len(record.model_mask_paths) > 1:
        second_path = str(list(record.model_mask_paths.values())[1])
    if not second_path:
        second_path = first_path
    first_gray = load_grayscale_image(Path(first_path))
    second_gray = load_grayscale_image(Path(second_path))
    if second_gray.shape != first_gray.shape:
        second_gray = resize_grayscale_image(second_gray, tuple(int(v) for v in first_gray.shape))
    base_gray = None
    base_path = record.base_path or record.original_path
    if base_path:
        base_gray = load_grayscale_image(Path(base_path))
        if base_gray.shape != first_gray.shape:
            base_gray = resize_grayscale_image(base_gray, tuple(int(v) for v in first_gray.shape))
    return {
        "first_gray": first_gray.copy(),
        "second_gray": second_gray.copy(),
        "first_binary": np.asarray(first_gray >= 128, dtype=bool),
        "second_binary": np.asarray(second_gray >= 128, dtype=bool),
        "base_gray": None if base_gray is None else base_gray.copy(),
        "shape": tuple(int(value) for value in first_gray.shape),
    }
