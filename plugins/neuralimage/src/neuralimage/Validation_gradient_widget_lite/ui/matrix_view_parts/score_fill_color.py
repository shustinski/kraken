"""matrix_view: score fill color."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor
from ...core.analysis_modes import metric_level_key, metric_visual_ratio
from ...core.domain import FrameRecord
from ...core.subpixel_grid import (
    SubpixelGridSpec,
)
from ...core.repository import metric_higher_is_better
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    MATRIX_BACKGROUND_ALT,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import (
    MIN_VISIBLE_TILE_SCREEN_SIZE,
    SUBPIXEL_VISIBILITY_THRESHOLD,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.compute_auto_color_window import (
    enhance_palette_position,
    interpolate_gradient_color,
    map_score_to_palette_position,
)

@dataclass(frozen=True, slots=True)
class MatrixLayoutConfig:
    """Describe the supported matrix layouts used by the lite matrix."""

    mode: str = "indexed_grid"
    total_frames: int = 0
    frames_per_row: int = 0
    rows: int = 1
    columns: int = 1

@dataclass(frozen=True, slots=True)
class MatrixTileSelection:
    """Describe one selected tile inside one matrix cell."""

    record: FrameRecord
    matrix_row: int
    matrix_column: int
    sub_row: int
    sub_column: int
    spec: SubpixelGridSpec
    parent_value: float
    subpixel_value: float
    subpixel_confidence: float | None = None
    aggregation: str = "mean"
    metric_key: str = "overall_frame_score"

def _tile_rect_for_cell(cell_rect: QRectF, tile_row: int, tile_column: int, spec: SubpixelGridSpec) -> QRectF:
    """Return one display tile rectangle inside a cell-local rectangle."""
    origin_x = float(cell_rect.left())
    origin_y = float(cell_rect.top())
    rows = max(1, int(spec.rows))
    columns = max(1, int(spec.columns))
    if tile_row < 0 or tile_column < 0 or tile_row >= rows or tile_column >= columns:
        return QRectF()
    width = max(0.0, float(cell_rect.width()))
    height = max(0.0, float(cell_rect.height()))
    if width <= 0.0 or height <= 0.0:
        return QRectF()
    left = width * (float(tile_column) / float(columns))
    top = height * (float(tile_row) / float(rows))
    right = width * (float(tile_column + 1) / float(columns))
    bottom = height * (float(tile_row + 1) / float(rows))
    tile_width = max(0.0, right - left)
    tile_height = max(0.0, bottom - top)
    if tile_width <= 0.0 or tile_height <= 0.0:
        return QRectF()
    return QRectF(origin_x + left, origin_y + top, tile_width, tile_height)

def _display_tile_index_for_cell(local_x: float, local_y: float, cell_rect: QRectF, spec: SubpixelGridSpec) -> tuple[int, int] | None:
    """Map one local point inside a matrix cell to the displayed subpixel index."""

    rows = max(1, int(spec.rows))
    columns = max(1, int(spec.columns))
    width = max(0.0, float(cell_rect.width()))
    height = max(0.0, float(cell_rect.height()))
    if width <= 0.0 or height <= 0.0:
        return None
    x = float(local_x)
    y = float(local_y)
    if x < 0.0 or y < 0.0 or x > width or y > height:
        return None
    normalized_x = min(max(x / width, 0.0), 1.0 - 1e-9)
    normalized_y = min(max(y / height, 0.0), 1.0 - 1e-9)
    column = int(normalized_x * float(columns))
    row = int(normalized_y * float(rows))
    return min(rows - 1, max(0, row)), min(columns - 1, max(0, column))

def _tile_screen_extent_for_rect(cell_rect: QRectF, spec: SubpixelGridSpec, zoom_level: float) -> float:
    """Estimate the visible screen size of one subpixel tile."""

    probe_rect = _tile_rect_for_cell(cell_rect, 0, 0, spec)
    if probe_rect.width() <= 0.0 or probe_rect.height() <= 0.0:
        return 0.0
    return min(float(probe_rect.width()), float(probe_rect.height())) * max(0.01, float(zoom_level))

def _subpixel_overlay_visible_for_rect(
    cell_rect: QRectF,
    spec: SubpixelGridSpec,
    zoom_level: float,
    *,
    zoom_threshold: float = SUBPIXEL_VISIBILITY_THRESHOLD,
) -> bool:
    """Return whether subpixel detail is worth drawing at the current zoom."""

    rows = max(1, int(spec.rows))
    columns = max(1, int(spec.columns))
    if rows * columns <= 1:
        return False
    normalized_zoom = max(0.01, float(zoom_level))
    if normalized_zoom < max(0.01, float(zoom_threshold)):
        return False
    return _tile_screen_extent_for_rect(cell_rect, spec, normalized_zoom) >= MIN_VISIBLE_TILE_SCREEN_SIZE

def _contrast_text_color(color: QColor) -> QColor:
    """Return a readable text color for one filled tile."""
    luminance = (0.299 * float(color.red()) + 0.587 * float(color.green()) + 0.114 * float(color.blue()))
    return QColor(20, 20, 20, 255) if luminance >= 150.0 else QColor(255, 255, 255, 255)

def _score_fill_color(
    metric_key: str | None,
    value: float,
    *,
    score_view_mode: str,
    gradient_name: str,
    point_match_radius: float,
    bce_score_cap: float,
) -> QColor:
    active_metric = str(metric_key or "")
    if score_view_mode == "absolute":
        ratio = metric_visual_ratio(
            active_metric,
            float(value),
            point_match_radius=float(point_match_radius),
            bce_score_cap=float(bce_score_cap),
        )
        if ratio is None:
            return QColor(MATRIX_BACKGROUND_ALT)
        level_key = metric_level_key(
            active_metric,
            float(value),
            point_match_radius=float(point_match_radius),
            bce_score_cap=float(bce_score_cap),
        )
        family = active_metric.split("::", 1)[0]
        higher_is_better = metric_higher_is_better(active_metric)
        if family == "model_confidence":
            if level_key == "score.level.low":
                return QColor(31, 95, 59, 235)
            if level_key == "score.level.moderate":
                return QColor(111, 122, 24, 235)
            if level_key == "score.level.elevated":
                return QColor(167, 93, 18, 235)
            return QColor(140, 47, 57, 235)
        if higher_is_better:
            if ratio < 0.33:
                return QColor(140, 47, 57, 235)
            if ratio < 0.66:
                return QColor(138, 106, 18, 235)
            return QColor(31, 95, 59, 235)
        if ratio < 0.33:
            return QColor(31, 95, 59, 235)
        if ratio < 0.66:
            return QColor(138, 106, 18, 235)
        return QColor(140, 47, 57, 235)
    position = map_score_to_palette_position(float(value), 0.0, 1.0)
    position = enhance_palette_position(position)
    return interpolate_gradient_color(gradient_name, position)

def _path_cache_identity(path_text: str | None) -> tuple[str, int, int]:
    text = str(path_text or "")
    if not text:
        return "", 0, 0
    try:
        path = Path(text)
        stat = path.stat()
        return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)
    except Exception:
        return text, 0, 0

def _record_path_signature(record: FrameRecord) -> tuple[object, ...]:
    model_masks = tuple(
        sorted((str(key), _path_cache_identity(str(value))) for key, value in (getattr(record, "model_mask_paths", {}) or {}).items())
    )
    model_probs = tuple(
        sorted((str(key), _path_cache_identity(str(value))) for key, value in (getattr(record, "model_prob_paths", {}) or {}).items())
    )
    return (
        _path_cache_identity(getattr(record, "first_path", None)),
        _path_cache_identity(getattr(record, "second_path", None)),
        _path_cache_identity(getattr(record, "original_path", None)),
        _path_cache_identity(getattr(record, "base_path", None)),
        model_masks,
        model_probs,
    )
