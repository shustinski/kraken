"""matrix_view: compute auto color window."""
from __future__ import annotations
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from PyQt6.QtGui import QColor
from ...core.backend_constants import FRAME_NUMBER_PATTERN
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    DEFAULT_ERROR_WINDOW,
    DEFAULT_GRADIENT_NAME,
    GRADIENT_PRESETS,
    NORMALIZATION_EPSILON,
)

@lru_cache(maxsize=65536)
def extract_frame_number(value: str) -> int:
    """Extract the zero-based frame number from the last underscore-separated filename segment."""
    stem = Path(str(value)).stem
    last_segment = stem.rsplit("_", 1)[-1]
    if not FRAME_NUMBER_PATTERN.fullmatch(last_segment):
        raise ValueError(f"Unable to extract frame number from '{value}'")
    return int(last_segment)

def blend_colors(base_color: QColor, overlay_color: QColor, alpha: float) -> QColor:
    """Blend two colors with the provided overlay alpha in the range 0..1."""
    ratio = max(0.0, min(float(alpha), 1.0))
    inv = 1.0 - ratio
    return QColor(
        int(base_color.red() * inv + overlay_color.red() * ratio),
        int(base_color.green() * inv + overlay_color.green() * ratio),
        int(base_color.blue() * inv + overlay_color.blue() * ratio),
    )

def interpolate_gradient_color(gradient_name: str, score: float) -> QColor:
    """Interpolate a QColor inside one named gradient preset."""
    preset = GRADIENT_PRESETS.get(gradient_name) or GRADIENT_PRESETS[DEFAULT_GRADIENT_NAME]
    value = max(0.0, min(float(score), 1.0))
    for index in range(1, len(preset)):
        left_pos, left_rgb = preset[index - 1]
        right_pos, right_rgb = preset[index]
        if value <= right_pos:
            span = max(NORMALIZATION_EPSILON, right_pos - left_pos)
            ratio = (value - left_pos) / span
            return QColor(
                int(left_rgb[0] + (right_rgb[0] - left_rgb[0]) * ratio),
                int(left_rgb[1] + (right_rgb[1] - left_rgb[1]) * ratio),
                int(left_rgb[2] + (right_rgb[2] - left_rgb[2]) * ratio),
            )
    return QColor(*preset[-1][1])

def error_palette_color(position: float, gradient_name: str = DEFAULT_GRADIENT_NAME) -> QColor:
    """Return one color from the named error gradient preset."""
    return interpolate_gradient_color(gradient_name, position)

def map_score_to_palette_position(score: float, low_bound: float, high_bound: float) -> float:
    """Map a score to the displayed gradient range, including inverted windows."""
    value = max(0.0, min(float(score), 1.0))
    low = max(0.0, min(float(low_bound), 1.0))
    high = max(0.0, min(float(high_bound), 1.0))
    if abs(high - low) < NORMALIZATION_EPSILON:
        return 1.0 if value >= high else 0.0
    if low < high:
        return max(0.0, min((value - low) / (high - low), 1.0))
    return max(0.0, min((low - value) / (low - high), 1.0))

def compute_auto_color_window(scores: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Derive one robust display window so small score changes stay visible."""
    if not scores:
        return DEFAULT_ERROR_WINDOW
    values = np.asarray(scores, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return DEFAULT_ERROR_WINDOW
    robust = np.sort(finite)
    if robust.size >= 4:
        diffs = np.diff(robust)
        positive_diffs = diffs[diffs > 0]
        median_gap = float(np.median(positive_diffs)) if positive_diffs.size > 0 else 0.0
        gap_threshold = max(0.05, median_gap * 6.0)
        start = 1 if float(diffs[0]) > gap_threshold else 0
        stop = robust.size - 1 if float(diffs[-1]) > gap_threshold else robust.size
        if stop - start >= 2:
            robust = robust[start:stop]
    low = float(np.quantile(robust, 0.08))
    high = float(np.quantile(robust, 0.92))
    if high <= low:
        return DEFAULT_ERROR_WINDOW
    min_span = 0.14
    finite_range = float(np.max(finite) - np.min(finite))
    if (high - low) < min_span and finite_range < min_span:
        center = float(np.median(finite))
        half = min_span * 0.5
        low = max(0.0, center - half)
        high = min(1.0, center + half)
        if (high - low) < min_span:
            if center <= 0.5:
                high = min(1.0, low + min_span)
            else:
                low = max(0.0, high - min_span)
    return max(0.0, low), min(1.0, high)

def enhance_palette_position(position: float) -> float:
    """Increase contrast near good/bad ends so weak variations are easier to spot."""
    value = max(0.0, min(float(position), 1.0))
    if value <= 0.5:
        return 0.5 * math.pow(value * 2.0, 0.82)
    return 1.0 - 0.5 * math.pow((1.0 - value) * 2.0, 0.82)
