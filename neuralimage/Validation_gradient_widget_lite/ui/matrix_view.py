"""Implement mismatch-only matrix layout, rendering and overview widgets for the lite tool."""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..core.backend_constants import FRAME_NUMBER_PATTERN
from ..core.analysis_modes import confidence_metric_family, metric_level_key, metric_visual_ratio
from ..core.domain import FrameRecord
from ..core.repository import (
    POLYGON_SUPPORT_THRESHOLD,
    _frame_uncertainty_components_from_probability,
    compute_comparison_score,
    load_frame_layers,
    load_grayscale_image,
)
from ..core.domain import ComparisonMode
from ..core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
    SubpixelSelection,
    aggregate_subpixel_values,
    build_subpixel_grid_from_array,
    build_subpixel_grid_from_pair,
)
from .i18n import Translator
from ..core.repository import metric_higher_is_better, natural_sort_key
from .ui_constants import (
    CARD_CONTENT_SPACING,
    DEFAULT_BORDER,
    DEFAULT_CELL_SIZE,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_GRADIENT_NAME,
    GRADIENT_LABELS,
    GRADIENT_PRESETS,
    GRADIENT_PREVIEW_MIN_HEIGHT,
    GRADIENT_RANGE_SELECTOR_MIN_HEIGHT,
    HOVER_BORDER,
    MATRIX_BACKGROUND,
    MATRIX_BACKGROUND_ALT,
    MATRIX_CELL_GAP,
    MATRIX_DEFAULT_PEN_WIDTH,
    MATRIX_HOVER_PEN_WIDTH,
    MATRIX_MAX_SCALE,
    MATRIX_MIN_CELL_SIZE,
    MATRIX_MIN_SCALE,
    MATRIX_PROCESSING_PEN_WIDTH,
    MATRIX_REFERENCE_PEN_WIDTH,
    MATRIX_SCENE_PADDING,
    MATRIX_SELECTED_BLEND_RATIO,
    MINIMAP_FRAME_MARGIN,
    MINIMAP_MIN_SIZE,
    MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH,
    MINIMAP_PROCESSING_TRIANGLE_HEIGHT,
    MINIMAP_REFERENCE_MARKER_SIDE,
    MINIMAP_REFERENCE_PEN_WIDTH,
    MINIMAP_SELECTED_COLOR,
    MINIMAP_SELECTED_OUTLINE_WIDTH,
    MINIMAP_SELECTED_RADIUS_OFF,
    MINIMAP_SELECTED_RADIUS_ON,
    NORMALIZATION_EPSILON,
    PANEL_BACKGROUND,
    PANEL_TEXT,
    PROCESSING_BORDER,
    PROCESSING_FILL,
    REFERENCE_BORDER,
    SELECTION_BLINK_INTERVAL_MS,
    SELECTED_BLINK_COLOR,
    SUBDUED_TEXT_COLOR,
    VISIBLE_RECT_MIN_SIZE,
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


SUBPIXEL_VISIBILITY_THRESHOLD = 2.50
MIN_VISIBLE_TILE_SCREEN_SIZE = 3.0
LOW_ZOOM_OVERVIEW_MAX_ZOOM = 1.35
LOW_ZOOM_OVERVIEW_RECORD_THRESHOLD = 20000
VIEW_LOD_OVERVIEW = "overview"
VIEW_LOD_PIXEL = "pixel"
VIEW_LOD_SUBPIXEL = "subpixel"
TILE_VIEWPORT_DEBOUNCE_MS = 90
TILE_LOAD_SLICE_BUDGET_MS = 12.0
TILE_LOAD_MAX_PER_SLICE = 2
TILE_PREFETCH_MARGIN_CELLS = 1
SUBPIXEL_GRID_CACHE_MAX_ITEMS = 2048
MATRIX_VIRTUALIZE_RECORD_THRESHOLD = 5000
MATRIX_ITEM_KEEP_MARGIN_CELLS = 2
MATRIX_MAX_MATERIALIZED_ITEMS = 2000


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


class _MatrixCellItem(QGraphicsRectItem):
    """Represent one visible matrix cell bound to one frame record."""

    def __init__(self, rect: QRectF, record: FrameRecord, row: int, column: int, index: int) -> None:
        super().__init__(rect)
        self.record = record
        self.row = int(row)
        self.column = int(column)
        self.index = int(index)
        self.subpixel_spec: SubpixelGridSpec | None = None
        self.subpixel_overlay_enabled = False
        self.subpixel_grid: SubpixelGrid | None = None
        self.subpixel_grid_provider = None
        self.subpixel_color_fn = None
        self.subpixel_metric_key: str | None = None
        self.selected_subpixel_selection: MatrixTileSelection | None = None
        self.hovered_subpixel_selection: MatrixTileSelection | None = None
        self._tile_rect_cache_key: tuple[float, float, float, float, int, int, str, int, int, int] | None = None
        self._tile_rect_cache: tuple[QRectF, ...] = ()

    def set_tile_state(
        self,
        selected_subpixel_selection: MatrixTileSelection | None,
        hovered_subpixel_selection: MatrixTileSelection | None,
    ) -> None:
        if (
            self.selected_subpixel_selection is selected_subpixel_selection
            and self.hovered_subpixel_selection is hovered_subpixel_selection
        ):
            return
        self.selected_subpixel_selection = selected_subpixel_selection
        self.hovered_subpixel_selection = hovered_subpixel_selection
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        spec = self.subpixel_grid.spec if self.subpixel_grid is not None else self.subpixel_spec
        if not self.subpixel_overlay_enabled or spec is None:
            super().paint(painter, option, widget)
            return
        zoom_level = max(0.01, abs(float(painter.worldTransform().m11())))
        rect = self.rect()
        if rect.width() < 6.0 or rect.height() < 6.0:
            super().paint(painter, option, widget)
            return
        rows = max(1, int(spec.rows))
        columns = max(1, int(spec.columns))
        if rows * columns <= 1:
            self._paint_base_cell(painter)
            return
        if not _subpixel_overlay_visible_for_rect(rect, spec, zoom_level):
            self._paint_base_cell(painter)
            return
        tile_screen_extent = _tile_screen_extent_for_rect(rect, spec, zoom_level)
        grid = self.subpixel_grid
        values = np.asarray(grid.values, dtype=np.float32) if grid is not None else None
        confidences = np.asarray(grid.confidences, dtype=np.float32) if grid is not None and grid.confidences is not None else None
        spec = grid.spec if grid is not None else spec
        if spec is None:
            super().paint(painter, option, widget)
            return
        if grid is None:
            self._paint_base_cell(painter)
            return
        if values is None:
            parent_score = float(self.record.score if bool(getattr(self.record, "score_ready", False)) else 0.0)
            rows = max(1, int(spec.rows))
            columns = max(1, int(spec.columns))
            values = np.full((rows, columns), parent_score, dtype=np.float32)
        else:
            rows = max(1, int(values.shape[0]))
            columns = max(1, int(values.shape[1]))
        selected_tile = self.selected_subpixel_selection
        hovered_tile = self.hovered_subpixel_selection
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        parent_fill = QColor(self.brush().color()) if self.brush().style() != Qt.BrushStyle.NoBrush else QColor(MATRIX_BACKGROUND_ALT)
        painter.fillRect(rect, parent_fill)
        painter.setPen(self.pen())
        painter.drawRect(rect)
        border_alpha = min(220, max(110, int(max(0.0, tile_screen_extent - MIN_VISIBLE_TILE_SCREEN_SIZE + 1.0) * 36.0)))
        grid_pen = QPen(QColor(92, 196, 255, border_alpha), 0.0)
        selected_pen = QPen(QColor(255, 228, 122, 235), 0.0)
        selected_pen.setCosmetic(True)
        hovered_pen = QPen(QColor(92, 196, 255, 215), 0.0)
        hovered_pen.setCosmetic(True)
        tile_rects = self._display_tile_rects(rect, spec, rows, columns)
        for tile_row in range(rows):
            for tile_column in range(columns):
                tile_rect = tile_rects[tile_row * columns + tile_column]
                if tile_rect.width() <= 0.0 or tile_rect.height() <= 0.0:
                    continue
                is_selected = (
                    selected_tile is not None
                    and selected_tile.matrix_row == self.row
                    and selected_tile.matrix_column == self.column
                    and selected_tile.sub_row == tile_row
                    and selected_tile.sub_column == tile_column
                )
                is_hovered = (
                    hovered_tile is not None
                    and hovered_tile.matrix_row == self.row
                    and hovered_tile.matrix_column == self.column
                    and hovered_tile.sub_row == tile_row
                    and hovered_tile.sub_column == tile_column
                )
                value = float(values[tile_row, tile_column])
                color_fn = self.subpixel_color_fn
                if is_selected:
                    fill = None
                    pen = selected_pen
                elif is_hovered:
                    fill = None
                    pen = hovered_pen
                else:
                    fill = color_fn(value) if callable(color_fn) else interpolate_gradient_color(DEFAULT_GRADIENT_NAME, max(0.0, min(value, 1.0)))
                    fill.setAlpha(220)
                    pen = grid_pen
                if fill is not None:
                    painter.fillRect(tile_rect, fill)
                painter.setPen(pen)
                painter.drawRect(tile_rect)
        if rows * columns > 1:
            painter.setPen(QPen(QColor(255, 255, 255, 60), 0.0))
            painter.drawRect(rect)
        painter.restore()

    def _paint_base_cell(self, painter: QPainter) -> None:
        rect = self.rect()
        painter.save()
        painter.fillRect(rect, self.brush())
        painter.setPen(self.pen())
        painter.drawRect(rect)
        painter.restore()

    def _display_tile_rects(self, rect: QRectF, spec: SubpixelGridSpec, rows: int, columns: int) -> tuple[QRectF, ...]:
        cache_key = (
            float(rect.left()),
            float(rect.top()),
            float(rect.width()),
            float(rect.height()),
            int(rows),
            int(columns),
            str(spec.mode),
            int(spec.tile_width),
            int(spec.tile_height),
            int(spec.overlap),
        )
        if self._tile_rect_cache_key == cache_key and len(self._tile_rect_cache) == rows * columns:
            return self._tile_rect_cache
        rects = tuple(
            _tile_rect_for_cell(rect, tile_row, tile_column, spec)
            for tile_row in range(rows)
            for tile_column in range(columns)
        )
        self._tile_rect_cache_key = cache_key
        self._tile_rect_cache = rects
        return rects


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
    low = float(np.quantile(finite, 0.08))
    high = float(np.quantile(finite, 0.92))
    if high <= low:
        return DEFAULT_ERROR_WINDOW
    min_span = 0.14
    if (high - low) < min_span:
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


def build_matrix_layout(records: list[FrameRecord], layout_config: MatrixLayoutConfig) -> tuple[list[tuple[FrameRecord, int, int]], int, int]:
    """Place frame records into one indexed or custom matrix layout."""
    if not records:
        return [], 0, 0
    mode = str(getattr(layout_config, "mode", "indexed_grid") or "indexed_grid")
    fixed_positions = all(
        record.identity is not None and record.identity.tile_x is not None and record.identity.tile_y is not None
        for record in records
    )
    if mode == "manual_grid":
        rows = int(layout_config.rows)
        columns = int(layout_config.columns)
        if rows <= 0 or columns <= 0:
            raise ValueError("Invalid custom matrix layout")
        capacity = rows * columns
        if len(records) > capacity:
            raise ValueError(f"Custom matrix layout capacity {capacity} is smaller than the frame count {len(records)}")
        if fixed_positions:
            placements: list[tuple[FrameRecord, int, int]] = []
            for record in records:
                row = int(record.identity.tile_y)
                column = int(record.identity.tile_x)
                if row < 0 or row >= rows or column < 0 or column >= columns:
                    raise ValueError("Stored matrix coordinates are outside the configured custom layout")
                placements.append((record, row, column))
            return sorted(placements, key=lambda item: (item[1], item[2], natural_sort_key(item[0].key))), columns, rows
        placements: list[tuple[FrameRecord, int, int]] = []
        for index, record in enumerate(records):
            row = index // columns
            column = index % columns
            placements.append((record, row, column))
        return placements, columns, rows

    columns = int(layout_config.frames_per_row)
    total_frames = int(layout_config.total_frames)
    if columns <= 0 or total_frames <= 0:
        raise ValueError("Invalid indexed matrix layout")
    if fixed_positions:
        rows = max(1, math.ceil(total_frames / columns))
        placements = []
        for record in records:
            row = int(record.identity.tile_y)
            column = int(record.identity.tile_x)
            if row < 0 or row >= rows or column < 0 or column >= columns:
                raise ValueError("Stored matrix coordinates are outside the configured indexed layout")
            placements.append((record, row, column))
        return sorted(placements, key=lambda item: (item[1], item[2], natural_sort_key(item[0].key))), columns, rows
    indexed_records: list[tuple[FrameRecord, int]] = []
    for record in sorted(records, key=lambda item: natural_sort_key(item.display_name or item.key)):
        frame_index = extract_frame_number(record.display_name or record.key)
        indexed_records.append((record, frame_index))
    if not indexed_records:
        return [], columns, 0
    raw_indices = [frame_index for _record, frame_index in indexed_records]
    min_index = min(raw_indices)
    max_index = max(raw_indices)
    one_based_sequential = min_index >= 1 and max_index <= total_frames and 0 not in raw_indices
    index_offset = 1 if one_based_sequential else 0
    normalized_max_index = max_index - index_offset
    if normalized_max_index < 0:
        raise ValueError("Invalid indexed matrix layout")
    effective_total_frames = max(total_frames, normalized_max_index + 1)
    rows = max(1, math.ceil(effective_total_frames / columns))
    placements: list[tuple[FrameRecord, int, int]] = []
    for record, raw_frame_index in indexed_records:
        frame_index = raw_frame_index - index_offset
        if frame_index < 0:
            raise ValueError(f"Frame index {raw_frame_index} is outside the configured matrix range")
        row = frame_index // columns
        column = frame_index % columns
        placements.append((record, row, column))
    return placements, columns, rows

class _GradientPreviewBar(QWidget):
    """Render a compact horizontal preview for one gradient preset."""

    def __init__(self, parent=None) -> None:
        """Initialize the preview bar."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self.setMinimumHeight(GRADIENT_PREVIEW_MIN_HEIGHT)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient shown in the preview bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the current gradient preview."""
        super().paintEvent(event)
        painter = QPainter(self)
        target = self.rect().adjusted(0, 2, 0, -2)
        if target.width() <= 0 or target.height() <= 0:
            painter.end()
            return
        for x in range(target.width()):
            position = x / max(1, target.width() - 1)
            painter.setPen(interpolate_gradient_color(self._gradient_name, position))
            painter.drawLine(target.left() + x, target.top(), target.left() + x, target.bottom())
        painter.setPen(QPen(SUBDUED_TEXT_COLOR, 1.0))
        painter.drawRect(target.adjusted(0, 0, -1, -1))
        painter.end()


class _GradientWindowBar(QWidget):
    """Render and edit the active error window on top of the current gradient."""

    rangeEdited = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        """Initialize the interactive error-window bar."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._low_bound, self._high_bound = DEFAULT_ERROR_WINDOW
        self._active_handle: str | None = None
        self.setMinimumHeight(max(26, GRADIENT_RANGE_SELECTOR_MIN_HEIGHT // 2))
        self.setMouseTracking(True)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient used by the range bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self.update()

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        """Update the current low and high bounds without emitting signals."""
        self._low_bound = max(0.0, min(float(low_bound), 1.0))
        self._high_bound = max(0.0, min(float(high_bound), 1.0))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the gradient and the current selected error window."""
        super().paintEvent(event)
        painter = QPainter(self)
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        if bar_rect.width() <= 0 or bar_rect.height() <= 0:
            painter.end()
            return
        for x in range(bar_rect.width()):
            position = x / max(1, bar_rect.width() - 1)
            painter.setPen(interpolate_gradient_color(self._gradient_name, position))
            painter.drawLine(bar_rect.left() + x, bar_rect.top(), bar_rect.left() + x, bar_rect.bottom())

        low_x = self._position_to_x(self._low_bound, bar_rect)
        high_x = self._position_to_x(self._high_bound, bar_rect)
        excluded_color = QColor(0, 0, 0, 110)
        if low_x < high_x:
            painter.fillRect(QRectF(bar_rect.left(), bar_rect.top(), max(0.0, low_x - bar_rect.left()), bar_rect.height()), excluded_color)
            painter.fillRect(QRectF(high_x, bar_rect.top(), max(0.0, bar_rect.right() - high_x), bar_rect.height()), excluded_color)
        elif low_x > high_x:
            painter.fillRect(QRectF(high_x, bar_rect.top(), max(0.0, low_x - high_x), bar_rect.height()), excluded_color)

        painter.setPen(QPen(PANEL_TEXT, 1.0))
        painter.drawRect(bar_rect.adjusted(0, 0, -1, -1))
        self._draw_handle(painter, low_x, bar_rect)
        self._draw_handle(painter, high_x, bar_rect)
        painter.end()

    def mousePressEvent(self, event) -> None:
        """Start dragging the nearest range handle."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        low_x = self._position_to_x(self._low_bound, bar_rect)
        high_x = self._position_to_x(self._high_bound, bar_rect)
        self._active_handle = 'low' if abs(event.position().x() - low_x) <= abs(event.position().x() - high_x) else 'high'
        self._update_active_handle(event.position().x(), bar_rect)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Update the active range handle while dragging."""
        if self._active_handle is None:
            super().mouseMoveEvent(event)
            return
        bar_rect = self.rect().adjusted(0, 6, 0, -8)
        self._update_active_handle(event.position().x(), bar_rect)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """Finish dragging the active range handle."""
        if event.button() == Qt.MouseButton.LeftButton and self._active_handle is not None:
            self._active_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _draw_handle(self, painter: QPainter, x_pos: float, bar_rect: QRectF) -> None:
        """Draw one vertical handle on the gradient bar."""
        painter.setPen(QPen(PANEL_TEXT, 2.0))
        painter.drawLine(QPointF(x_pos, bar_rect.top() - 3), QPointF(x_pos, bar_rect.bottom() + 3))

    def _position_to_x(self, value: float, bar_rect: QRectF) -> float:
        """Convert a normalized value into a horizontal bar coordinate."""
        return bar_rect.left() + max(0.0, min(float(value), 1.0)) * max(1.0, bar_rect.width() - 1.0)

    def _x_to_position(self, x_pos: float, bar_rect: QRectF) -> float:
        """Convert a horizontal bar coordinate into a normalized value."""
        if bar_rect.width() <= 1.0:
            return 0.0
        return max(0.0, min((x_pos - bar_rect.left()) / (bar_rect.width() - 1.0), 1.0))

    def _update_active_handle(self, x_pos: float, bar_rect: QRectF) -> None:
        """Update the active handle and emit the new selected range."""
        position = self._x_to_position(x_pos, bar_rect)
        if self._active_handle == 'low':
            self._low_bound = position
        elif self._active_handle == 'high':
            self._high_bound = position
        self.update()
        self.rangeEdited.emit(self._low_bound, self._high_bound)


class _GradientPresetCard(QFrame):
    """Render one clickable gradient preset card."""

    clicked = pyqtSignal(str)

    def __init__(self, gradient_name: str, parent=None) -> None:
        """Initialize one gradient preset card."""
        super().__init__(parent)
        self._gradient_name = gradient_name
        self._selected = False
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(CARD_CONTENT_SPACING)
        self._label = QLabel(GRADIENT_LABELS.get(gradient_name, gradient_name.title()), self)
        self._preview = _GradientPreviewBar(self)
        self._preview.set_gradient_name(gradient_name)
        layout.addWidget(self._label)
        layout.addWidget(self._preview)
        self._refresh_style()

    def gradient_name(self) -> str:
        """Return the gradient preset name represented by this card."""
        return self._gradient_name

    def set_selected(self, selected: bool) -> None:
        """Update the selected state of the card."""
        self._selected = bool(selected)
        self._refresh_style()

    def mousePressEvent(self, event) -> None:
        """Emit the card click when the user presses the left mouse button."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._gradient_name)
            event.accept()
            return
        super().mousePressEvent(event)

    def _refresh_style(self) -> None:
        """Refresh the card border according to the selected state."""
        border_color = PANEL_TEXT if self._selected else SUBDUED_TEXT_COLOR
        background_color = '#2f2f31' if self._selected else '#262628'
        self.setStyleSheet(
            f'QFrame {{ border: 1px solid {border_color.name()}; border-radius: 4px; background: {background_color}; }}'
            'QLabel { border: none; background: transparent; color: #f0f0f0; }'
        )


class GradientPresetSelectorWidget(QWidget):
    """Select the active gradient preset used to render matrix scores."""

    gradientChanged = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        """Initialize the gradient preset selector."""
        super().__init__(parent)
        self._selected_gradient_name = DEFAULT_GRADIENT_NAME
        self._cards: dict[str, _GradientPresetCard] = {}
        self._i18n = Translator()
        self._t = self._i18n.tr
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_CONTENT_SPACING)
        self._title_label = QLabel(self._t("matrix.gradient"), self)
        layout.addWidget(self._title_label)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(CARD_CONTENT_SPACING)
        grid.setVerticalSpacing(CARD_CONTENT_SPACING)
        for index, name in enumerate(GRADIENT_PRESETS):
            card = _GradientPresetCard(name, self)
            card.clicked.connect(self._on_card_clicked)
            self._cards[name] = card
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)
        self.set_selected_gradient(DEFAULT_GRADIENT_NAME, emit_signal=False)

    def selected_gradient(self) -> str:
        """Return the currently selected gradient preset name."""
        return self._selected_gradient_name

    def set_selected_gradient(self, name: str, *, emit_signal: bool = True) -> None:
        """Select one gradient preset and refresh the visible card states."""
        normalized = name if name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        previous = self._selected_gradient_name
        self._selected_gradient_name = normalized
        for gradient_name, card in self._cards.items():
            card.set_selected(gradient_name == normalized)
        if emit_signal and normalized != previous:
            self.gradientChanged.emit(normalized)

    def _on_card_clicked(self, gradient_name: str) -> None:
        """Handle one direct click on a gradient preset card."""
        self.set_selected_gradient(gradient_name, emit_signal=True)

    def retranslate_ui(self) -> None:
        """Update translated captions inside the gradient selector."""
        self._title_label.setText(self._t("matrix.gradient"))


class GradientRangeSelectorWidget(QWidget):
    """Edit the low and high bounds of the active error gradient window."""

    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        """Initialize the error-window selector."""
        super().__init__(parent)
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._i18n = Translator()
        self._t = self._i18n.tr
        self._low_spin = QDoubleSpinBox(self)
        self._high_spin = QDoubleSpinBox(self)
        self._bar = _GradientWindowBar(self)
        for spin, value in ((self._low_spin, DEFAULT_ERROR_WINDOW[0]), (self._high_spin, DEFAULT_ERROR_WINDOW[1])):
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(2)
            spin.setValue(value)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_CONTENT_SPACING)
        self._title_label = QLabel(self._t("matrix.error_window"), self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._bar)
        row = QHBoxLayout()
        self._low_label = QLabel(self._t("matrix.low"), self)
        row.addWidget(self._low_label)
        row.addWidget(self._low_spin)
        self._high_label = QLabel(self._t("matrix.high"), self)
        row.addWidget(self._high_label)
        row.addWidget(self._high_spin)
        layout.addLayout(row)
        self.setMinimumHeight(GRADIENT_RANGE_SELECTOR_MIN_HEIGHT)
        self._low_spin.valueChanged.connect(self._emit_range_changed)
        self._high_spin.valueChanged.connect(self._emit_range_changed)
        self._bar.rangeEdited.connect(self._on_bar_range_edited)
        self._bar.set_gradient_name(self._gradient_name)
        self._bar.set_error_window(*DEFAULT_ERROR_WINDOW)

    def error_window(self) -> tuple[float, float]:
        """Return the currently selected error window."""
        return float(self._low_spin.value()), float(self._high_spin.value())

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        """Set the error window without emitting change signals."""
        low_value = max(0.0, min(float(low_bound), 1.0))
        high_value = max(0.0, min(float(high_bound), 1.0))
        self._low_spin.blockSignals(True)
        self._high_spin.blockSignals(True)
        self._low_spin.setValue(low_value)
        self._high_spin.setValue(high_value)
        self._low_spin.blockSignals(False)
        self._high_spin.blockSignals(False)
        self._bar.set_error_window(low_value, high_value)

    def set_gradient_name(self, gradient_name: str) -> None:
        """Set the gradient used by the error-window preview bar."""
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME
        self._bar.set_gradient_name(self._gradient_name)

    def _emit_range_changed(self, _value: float) -> None:
        """Propagate range changes coming from the numeric editors."""
        low_value, high_value = self.error_window()
        self._bar.set_error_window(low_value, high_value)
        self.rangeChanged.emit(low_value, high_value)

    def _on_bar_range_edited(self, low_bound: float, high_bound: float) -> None:
        """Apply range changes coming from the draggable gradient bar."""
        self.set_error_window(low_bound, high_bound)
        self.rangeChanged.emit(*self.error_window())

    def retranslate_ui(self) -> None:
        """Update translated captions inside the range selector."""
        self._title_label.setText(self._t("matrix.error_window"))
        self._low_label.setText(self._t("matrix.low"))
        self._high_label.setText(self._t("matrix.high"))


class MatrixMiniMapWidget(QWidget):
    """Render a compact overview image for the active matrix tab."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._visible_rect = QRectF()
        self._selected_position: tuple[int, int] | None = None
        self._selected_blink_on = False
        self._processing_positions: tuple[tuple[int, int], ...] = ()
        self._reference_position: tuple[int, int] | None = None
        self.setMinimumSize(*MINIMAP_MIN_SIZE)

    def set_overview(self, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position) -> None:
        self._image = image
        self._visible_rect = QRectF(visible_rect)
        self._selected_position = selected_position
        self._selected_blink_on = bool(selected_blink_on)
        self._processing_positions = tuple(processing_positions)
        self._reference_position = reference_position
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), PANEL_BACKGROUND)
        if self._image is None or self._image.isNull():
            painter.setPen(PANEL_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._t("matrix.no_matrix"))
            painter.end()
            return
        target = self.rect().adjusted(MINIMAP_FRAME_MARGIN, MINIMAP_FRAME_MARGIN, -MINIMAP_FRAME_MARGIN, -MINIMAP_FRAME_MARGIN)
        painter.drawImage(target, self._image)
        width = max(1, self._image.width())
        height = max(1, self._image.height())
        cell_w = target.width() / width
        cell_h = target.height() / height
        if not self._visible_rect.isNull():
            rect = QRectF(
                target.left() + self._visible_rect.left() * target.width(),
                target.top() + self._visible_rect.top() * target.height(),
                max(VISIBLE_RECT_MIN_SIZE, self._visible_rect.width() * target.width()),
                max(VISIBLE_RECT_MIN_SIZE, self._visible_rect.height() * target.height()),
            )
            painter.setPen(QPen(PANEL_TEXT, 1.5))
            painter.drawRect(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PROCESSING_FILL)
        for row, column in self._processing_positions:
            painter.drawEllipse(
                QPointF(target.left() + (column + 0.5) * cell_w, target.top() + (row + 0.5) * cell_h),
                MINIMAP_PROCESSING_TRIANGLE_HALF_WIDTH,
                MINIMAP_PROCESSING_TRIANGLE_HEIGHT,
            )
        if self._reference_position is not None:
            row, column = self._reference_position
            painter.setPen(QPen(REFERENCE_BORDER, MINIMAP_REFERENCE_PEN_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(
                    target.left() + column * cell_w,
                    target.top() + row * cell_h,
                    max(MINIMAP_REFERENCE_MARKER_SIDE, cell_w),
                    max(MINIMAP_REFERENCE_MARKER_SIDE, cell_h),
                )
            )
        if self._selected_position is not None:
            row, column = self._selected_position
            radius = MINIMAP_SELECTED_RADIUS_ON if self._selected_blink_on else MINIMAP_SELECTED_RADIUS_OFF
            painter.setPen(QPen(MINIMAP_SELECTED_COLOR, MINIMAP_SELECTED_OUTLINE_WIDTH))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPointF(target.left() + (column + 0.5) * cell_w, target.top() + (row + 0.5) * cell_h),
                radius,
                radius,
            )
        painter.end()


class MatrixListWidget(QGraphicsView):
    """Render the frame matrix and provide navigation, selection and overview data."""
    recordActivated = pyqtSignal(object)
    recordSelected = pyqtSignal(object)
    tileSelected = pyqtSignal(object)
    tileActivated = pyqtSignal(object)
    contextMenuRequested = pyqtSignal(object, object)
    overviewChanged = pyqtSignal(object, object, object, object, object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._i18n = Translator()
        self._t = self._i18n.tr
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._records: list[FrameRecord] = []
        self._cell_size = DEFAULT_CELL_SIZE
        self._gap = MATRIX_CELL_GAP
        self._scene_padding = MATRIX_SCENE_PADDING
        self._selected_item: _MatrixCellItem | None = None
        self._hovered_item: _MatrixCellItem | None = None
        self._selected_subpixel_selection: MatrixTileSelection | None = None
        self._hovered_subpixel_selection: MatrixTileSelection | None = None
        self._processing_keys: set[str] = set()
        self._reference_key: str | None = None
        self._columns = 0
        self._rows = 0
        self._gradient_name = DEFAULT_GRADIENT_NAME
        self._error_window_low, self._error_window_high = DEFAULT_ERROR_WINDOW
        self._auto_color_window_low, self._auto_color_window_high = DEFAULT_ERROR_WINDOW
        self._score_view_mode = "relative"
        self._metric_key: str | None = None
        self._point_match_radius = 3.0
        self._bce_score_cap = 1.0
        self._overview_image: QImage | None = None
        self._selection_blink_on = False
        self._selection_blink_timer = QTimer(self)
        self._selection_blink_timer.setInterval(SELECTION_BLINK_INTERVAL_MS)
        self._selection_blink_timer.timeout.connect(self._toggle_selection_blink)
        self._selection_blink_timer.start()
        self._layout_config = MatrixLayoutConfig()
        self._subpixel_spec: SubpixelGridSpec | None = None
        self._subpixel_aggregation = "mean"
        self._subpixel_comparison_mode = ComparisonMode.DISAGREEMENT
        self._subpixel_grid_cache: OrderedDict[tuple[object, ...], SubpixelGrid] = OrderedDict()
        self._item_by_key: dict[str, _MatrixCellItem] = {}
        self._record_by_position: dict[tuple[int, int], FrameRecord] = {}
        self._record_positions: dict[str, tuple[int, int]] = {}
        self._record_index_by_key: dict[str, int] = {}
        self._overview_layer_item: QGraphicsPixmapItem | None = None
        self._matrix_frame_item: QGraphicsRectItem | None = None
        self._virtualized_items_enabled = False
        self._tile_zoom_threshold = SUBPIXEL_VISIBILITY_THRESHOLD
        self._tile_overlay_visible = False
        self._overview_layer_visible = False
        self._active_lod_band = VIEW_LOD_PIXEL
        self._tile_request_generation = 0
        self._tile_load_generation: int | None = None
        self._pending_tile_keys: list[str] = []
        self._pending_tile_key_set: set[str] = set()
        self._tile_viewport_timer = QTimer(self)
        self._tile_viewport_timer.setSingleShot(True)
        self._tile_viewport_timer.setInterval(TILE_VIEWPORT_DEBOUNCE_MS)
        self._tile_viewport_timer.timeout.connect(self._prepare_visible_tile_queue)
        self._tile_load_timer = QTimer(self)
        self._tile_load_timer.setSingleShot(True)
        self._tile_load_timer.timeout.connect(self._process_pending_tile_queue)
        self._pan_active = False
        self._pan_start = None
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(MATRIX_BACKGROUND)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.horizontalScrollBar().valueChanged.connect(self._on_viewport_scroll_changed)
        self.verticalScrollBar().valueChanged.connect(self._on_viewport_scroll_changed)

    def _clear_subpixel_grid_cache(self) -> None:
        self._subpixel_grid_cache.clear()

    def _clear_record_subpixel_grids(self) -> None:
        for record in self._records:
            try:
                record.subpixel_grid = None
            except Exception:
                pass

    def _subpixel_cache_get(self, cache_key: tuple[object, ...]) -> SubpixelGrid | None:
        cached = self._subpixel_grid_cache.get(cache_key)
        if cached is not None:
            self._subpixel_grid_cache.move_to_end(cache_key)
        return cached

    def _subpixel_cache_put(self, cache_key: tuple[object, ...], grid: SubpixelGrid) -> None:
        self._subpixel_grid_cache[cache_key] = grid
        self._subpixel_grid_cache.move_to_end(cache_key)
        while len(self._subpixel_grid_cache) > SUBPIXEL_GRID_CACHE_MAX_ITEMS:
            self._subpixel_grid_cache.popitem(last=False)

    @staticmethod
    def _record_path_signature(record: FrameRecord) -> tuple[object, ...]:
        model_masks = tuple(sorted((str(key), str(value)) for key, value in (getattr(record, "model_mask_paths", {}) or {}).items()))
        model_probs = tuple(sorted((str(key), str(value)) for key, value in (getattr(record, "model_prob_paths", {}) or {}).items()))
        return (
            str(getattr(record, "first_path", None) or ""),
            str(getattr(record, "second_path", None) or ""),
            str(getattr(record, "original_path", None) or ""),
            str(getattr(record, "base_path", None) or ""),
            model_masks,
            model_probs,
        )

    def _subpixel_cache_key_for_record(self, record: FrameRecord, spec: SubpixelGridSpec) -> tuple[object, ...]:
        return (
            str(record.key),
            str(spec.mode),
            int(spec.rows),
            int(spec.columns),
            int(spec.tile_width),
            int(spec.tile_height),
            int(spec.overlap),
            str(self._subpixel_aggregation),
            str(self._metric_key or ""),
            str(getattr(self._subpixel_comparison_mode, "value", self._subpixel_comparison_mode)),
            self._record_path_signature(record),
        )

    def _on_viewport_scroll_changed(self, *_args) -> None:
        self._emit_overview_state()
        self._sync_visible_matrix_items()
        self._schedule_visible_tile_request()

    def set_cell_size(self, cell_size: int) -> None:
        self._cell_size = max(MATRIX_MIN_CELL_SIZE, int(cell_size))

    def set_gradient_preset(self, gradient_name: str) -> None:
        self._gradient_name = gradient_name if gradient_name in GRADIENT_PRESETS else DEFAULT_GRADIENT_NAME

    def set_error_window(self, low_bound: float, high_bound: float) -> None:
        self._error_window_low = max(0.0, min(float(low_bound), 1.0))
        self._error_window_high = max(0.0, min(float(high_bound), 1.0))

    def set_score_view_mode(self, mode: str | None) -> None:
        normalized = str(mode or "relative").strip().lower()
        self._score_view_mode = "absolute" if normalized == "absolute" else "relative"

    def set_metric_context(self, metric_key: str | None, *, point_match_radius: float, bce_score_cap: float) -> None:
        previous_metric_key = self._metric_key
        self._metric_key = None if metric_key is None else str(metric_key)
        self._point_match_radius = float(point_match_radius)
        self._bce_score_cap = float(bce_score_cap)
        if self._metric_key != previous_metric_key:
            self._clear_subpixel_grid_cache()
            self._clear_record_subpixel_grids()
            self._invalidate_tile_requests()
        for item in self._item_by_key.values():
            item.subpixel_metric_key = self._metric_key
            if self._metric_key != previous_metric_key:
                item.subpixel_grid = None
                item.update()

    def set_layout_config(self, layout_config: MatrixLayoutConfig) -> None:
        self._layout_config = layout_config

    def set_subpixel_grid_spec(self, spec: SubpixelGridSpec | None, *, aggregation: str = "mean") -> None:
        previous_spec = self._subpixel_spec
        previous_aggregation = self._subpixel_aggregation
        self._subpixel_spec = None if spec is None else spec.normalized()
        self._subpixel_aggregation = str(aggregation or "mean")
        if self._subpixel_spec == previous_spec and self._subpixel_aggregation == previous_aggregation:
            return
        cache_invalidated = self._subpixel_spec != previous_spec or self._subpixel_aggregation != previous_aggregation
        if self._subpixel_spec != previous_spec:
            self._selected_subpixel_selection = None
            self._hovered_subpixel_selection = None
        if cache_invalidated:
            self._clear_subpixel_grid_cache()
            self._clear_record_subpixel_grids()
            self._invalidate_tile_requests()
        for item in self._item_by_key.values():
            item.subpixel_spec = self._subpixel_spec
            item.subpixel_overlay_enabled = self._subpixel_spec is not None
            item.subpixel_grid_provider = self._subpixel_grid_for_record if self._subpixel_spec is not None else None
            item.subpixel_color_fn = self._subpixel_color_for_value
            item.subpixel_metric_key = self._metric_key
            item.subpixel_grid = None
            item.update()
        if self._subpixel_spec != previous_spec:
            self._sync_tile_state_for_keys(self._item_by_key.keys())
        self._update_tile_lod(force=True)

    def set_subpixel_comparison_mode(self, comparison_mode) -> None:
        previous_mode = self._subpixel_comparison_mode
        if isinstance(comparison_mode, ComparisonMode):
            self._subpixel_comparison_mode = comparison_mode
        else:
            try:
                self._subpixel_comparison_mode = ComparisonMode(str(comparison_mode))
            except Exception:
                self._subpixel_comparison_mode = ComparisonMode.DISAGREEMENT
        if self._subpixel_comparison_mode != previous_mode:
            self._clear_subpixel_grid_cache()
            self._clear_record_subpixel_grids()
            self._invalidate_tile_requests()
            for item in self._item_by_key.values():
                item.subpixel_grid = None
                item.update()

    def set_tile_grid_plan(self, plan) -> None:  # pragma: no cover - compatibility shim
        if plan is None:
            self.set_subpixel_grid_spec(None)
            return
        rows = int(getattr(plan, "rows", 0) or 0)
        columns = int(getattr(plan, "columns", 0) or 0)
        self.set_subpixel_grid_spec(
            SubpixelGridSpec.from_tile_plan(
                tile_width=int(getattr(plan, "tile_width", 1) or 1),
                tile_height=int(getattr(plan, "tile_height", 1) or 1),
                overlap=int(getattr(plan, "overlap", 0) or 0),
                rows=rows,
                columns=columns,
            )
        )

    def _subpixel_grid_for_record(self, record: FrameRecord) -> SubpixelGrid | None:
        spec = self._subpixel_spec
        if spec is None:
            return getattr(record, "subpixel_grid", None)
        cache_key = self._subpixel_cache_key_for_record(record, spec)
        cached = self._subpixel_cache_get(cache_key)
        if cached is not None:
            return cached
        metric_family = confidence_metric_family(self._metric_key)
        if metric_family is not None:
            family, model_id = metric_family
            probability_path = str((getattr(record, "model_prob_paths", {}) or {}).get(model_id) or "")
            probability_array = None
            try:
                if probability_path:
                    probability_array = np.asarray(load_grayscale_image(Path(probability_path)), dtype=np.float32) / 255.0
                elif model_id and model_id in (getattr(record, "model_mask_paths", {}) or {}):
                    mask_path = str((getattr(record, "model_mask_paths", {}) or {}).get(model_id) or "")
                    if mask_path:
                        probability_array = np.asarray(load_grayscale_image(Path(mask_path)), dtype=np.float32) / 255.0
            except Exception:
                probability_array = None
            if probability_array is not None and probability_array.ndim == 2 and probability_array.size > 0:
                try:
                    if family == "model_confidence":
                        grid = build_subpixel_grid_from_array(
                            probability_array,
                            spec,
                            score_fn=lambda prob_tile: _frame_uncertainty_components_from_probability(
                                np.asarray(prob_tile, dtype=np.float32),
                                support_threshold=float(POLYGON_SUPPORT_THRESHOLD),
                            )[0],
                            aggregation=self._subpixel_aggregation,
                            value_kind="risk",
                        )
                    elif family == "model_uncertain_fraction":
                        grid = build_subpixel_grid_from_array(
                            probability_array,
                            spec,
                            score_fn=lambda prob_tile: _frame_uncertainty_components_from_probability(
                                np.asarray(prob_tile, dtype=np.float32),
                                support_threshold=float(POLYGON_SUPPORT_THRESHOLD),
                            )[2],
                            aggregation=self._subpixel_aggregation,
                            value_kind="risk",
                        )
                    else:
                        grid = None
                    if grid is not None:
                        self._subpixel_cache_put(cache_key, grid)
                        try:
                            record.subpixel_grid = grid
                        except Exception:
                            pass
                        return grid
                except Exception:
                    pass
        try:
            layers = load_frame_layers(record)
            first_layer = np.asarray(layers.get("first_binary"), dtype=bool)
            second_layer = np.asarray(layers.get("second_binary"), dtype=bool)
            if first_layer.shape == second_layer.shape and first_layer.ndim == 2 and first_layer.size > 0:
                grid = build_subpixel_grid_from_pair(
                    first_layer,
                    second_layer,
                    spec,
                    score_fn=lambda first_tile, second_tile: compute_comparison_score(first_tile, second_tile, self._subpixel_comparison_mode),
                    aggregation=self._subpixel_aggregation,
                    value_kind="risk",
                )
                self._subpixel_cache_put(cache_key, grid)
                try:
                    record.subpixel_grid = grid
                except Exception:
                    pass
                return grid
        except Exception:
            pass
        parent_score = float(record.score if bool(getattr(record, "score_ready", False)) else 0.0)
        values = np.full((max(1, int(spec.rows)), max(1, int(spec.columns))), parent_score, dtype=np.float32)
        confidences = np.ones_like(values, dtype=np.float32)
        grid = SubpixelGrid(spec=spec.normalized(), values=values, confidences=confidences, aggregation=self._subpixel_aggregation, value_kind="score")
        self._subpixel_cache_put(cache_key, grid)
        try:
            record.subpixel_grid = grid
        except Exception:
            pass
        return grid


    def set_processing_keys(self, processing_keys) -> None:
        previous = set(self._processing_keys)
        self._processing_keys = {str(key) for key in processing_keys}
        for key in previous | self._processing_keys:
            item = self._ensure_item_for_key(key) if key in self._processing_keys else self._item_by_key.get(key)
            if item is not None:
                self._apply_item_style(item)
        self._sync_visible_matrix_items()
        self._sync_low_zoom_visibility_for_keys(previous | self._processing_keys)
        self._emit_overview_state()

    def set_reference_key(self, reference_key: str | None) -> None:
        previous = self._reference_key
        self._reference_key = str(reference_key) if reference_key else None
        for key in {previous, self._reference_key}:
            item = self._ensure_item_for_key(key) if key == self._reference_key else self._item_by_key.get(str(key)) if key else None
            if item is not None:
                self._apply_item_style(item)
        self._sync_visible_matrix_items()
        self._sync_low_zoom_visibility_for_keys({previous, self._reference_key})
        self._emit_overview_state()

    def set_records(self, records: list[FrameRecord], *, sort_mode: str = "name", reset_view: bool = False) -> None:
        if sort_mode == "name" and str(getattr(self._layout_config, "mode", "indexed_grid") or "indexed_grid") == "indexed_grid":
            ordered = list(records)
        else:
            ordered = self._sort_records(list(records), sort_mode)
        if reset_view:
            self.resetTransform()
        self._rebuild_scene(ordered)

    def refresh_scene(self) -> None:
        if not self._records:
            return
        ordered_items = sorted(self._item_by_key.values(), key=lambda item: item.index)
        for item in ordered_items:
            item.setToolTip(self._tooltip_for_record(item.record))
            self._apply_item_style(item, sync_tile_state=False)
        self._sync_tile_state_for_keys(self._item_by_key.keys())
        placements = [(item.record, item.row, item.column) for item in ordered_items]
        self._overview_image = self._build_overview_image(placements)
        self._refresh_overview_layer_pixmap()
        self._scene.update()
        self.viewport().update()
        self._emit_overview_state()
        self._update_tile_lod(force=True)

    def current_record(self) -> FrameRecord | None:
        return self._selected_item.record if self._selected_item is not None else None

    def selected_tile_selection(self) -> MatrixTileSelection | None:
        return self._selected_subpixel_selection

    def select_record_by_key(self, key: str, *, ensure_visible: bool = True) -> FrameRecord | None:
        item = self._ensure_item_for_key(str(key))
        if item is None:
            return None
        self._select_item(item)
        self.recordSelected.emit(item.record)
        if ensure_visible:
            self.centerOn(item)
            self._emit_overview_state()
        return item.record

    def neighbor_record(self, record: FrameRecord | str, direction: str) -> FrameRecord | None:
        key = record.key if isinstance(record, FrameRecord) else str(record)
        position = self._record_positions.get(key)
        if position is None:
            return None
        row, column = position
        direction_name = str(direction).lower()
        if direction_name == "right":
            return self._next_horizontal_record(row, column, step=1)
        if direction_name == "left":
            return self._next_horizontal_record(row, column, step=-1)
        if direction_name == "down":
            return self._next_vertical_record(row, column, step=1)
        if direction_name == "up":
            return self._next_vertical_record(row, column, step=-1)
        return None

    def _next_horizontal_record(self, row: int, column: int, *, step: int) -> FrameRecord | None:
        row_columns = sorted(col for (item_row, col) in self._record_by_position if item_row == row)
        if step > 0:
            for candidate_col in row_columns:
                if candidate_col > column:
                    return self._record_by_position.get((row, candidate_col))
            for candidate_row in sorted({item_row for (item_row, _col) in self._record_by_position if item_row > row}):
                candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
                if candidate_cols:
                    return self._record_by_position.get((candidate_row, candidate_cols[0]))
            return None
        for candidate_col in reversed(row_columns):
            if candidate_col < column:
                return self._record_by_position.get((row, candidate_col))
        for candidate_row in sorted({item_row for (item_row, _col) in self._record_by_position if item_row < row}, reverse=True):
            candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
            if candidate_cols:
                return self._record_by_position.get((candidate_row, candidate_cols[-1]))
        return None

    def _next_vertical_record(self, row: int, column: int, *, step: int) -> FrameRecord | None:
        rows = sorted({item_row for (item_row, _col) in self._record_by_position if item_row != row})
        if step < 0:
            rows = list(reversed(rows))
        for candidate_row in rows:
            if step > 0 and candidate_row <= row:
                continue
            if step < 0 and candidate_row >= row:
                continue
            if (candidate_row, column) in self._record_by_position:
                return self._record_by_position[(candidate_row, column)]
            candidate_cols = sorted(col for (item_row, col) in self._record_by_position if item_row == candidate_row)
            if candidate_cols:
                nearest_col = min(candidate_cols, key=lambda value: abs(value - column))
                return self._record_by_position.get((candidate_row, nearest_col))
        return None

    def _invalidate_tile_requests(self) -> None:
        self._tile_request_generation += 1
        self._tile_load_generation = None
        self._pending_tile_keys.clear()
        self._pending_tile_key_set.clear()
        self._tile_viewport_timer.stop()
        self._tile_load_timer.stop()

    def _lod_band_for_current_view(self) -> str:
        if self._overview_layer_should_be_active():
            return VIEW_LOD_OVERVIEW
        if self._subpixel_overlay_visible():
            return VIEW_LOD_SUBPIXEL
        return VIEW_LOD_PIXEL

    def _schedule_visible_tile_request(self, *, immediate: bool = False) -> None:
        if self._subpixel_spec is None or not self._tile_overlay_visible:
            self._invalidate_tile_requests()
            return
        self._tile_request_generation += 1
        self._tile_load_generation = None
        self._pending_tile_keys.clear()
        self._pending_tile_key_set.clear()
        self._tile_load_timer.stop()
        if immediate:
            self._tile_viewport_timer.stop()
            self._prepare_visible_tile_queue()
        else:
            self._tile_viewport_timer.start()

    def _visible_record_keys(self, *, margin_cells: int = TILE_PREFETCH_MARGIN_CELLS) -> tuple[str, ...]:
        if not self._record_by_position or self._columns <= 0 or self._rows <= 0:
            return tuple()
        visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        if visible_scene_rect.width() <= 0.0 or visible_scene_rect.height() <= 0.0:
            return tuple()
        span = float(self._cell_size + self._gap)
        if span <= 0.0:
            return tuple()
        margin = max(0, int(margin_cells))
        left = float(visible_scene_rect.left()) - float(self._scene_padding)
        right = float(visible_scene_rect.right()) - float(self._scene_padding)
        top = float(visible_scene_rect.top()) - float(self._scene_padding)
        bottom = float(visible_scene_rect.bottom()) - float(self._scene_padding)
        min_column = max(0, int(math.floor(left / span)) - margin)
        max_column = min(self._columns - 1, int(math.floor(right / span)) + margin)
        min_row = max(0, int(math.floor(top / span)) - margin)
        max_row = min(self._rows - 1, int(math.floor(bottom / span)) + margin)
        if min_column > max_column or min_row > max_row:
            return tuple()
        center_row = (min_row + max_row) * 0.5
        center_column = (min_column + max_column) * 0.5
        candidates: list[tuple[float, str]] = []
        for row in range(min_row, max_row + 1):
            for column in range(min_column, max_column + 1):
                record = self._record_by_position.get((row, column))
                if record is None:
                    continue
                distance = abs(float(row) - center_row) + abs(float(column) - center_column)
                candidates.append((distance, str(record.key)))
        candidates.sort(key=lambda item: (item[0], natural_sort_key(item[1])))
        return tuple(key for _distance, key in candidates)

    def _assign_cached_subpixel_grid(self, item: _MatrixCellItem) -> bool:
        spec = self._subpixel_spec
        if spec is None:
            return False
        cached = self._subpixel_cache_get(self._subpixel_cache_key_for_record(item.record, spec))
        if cached is None:
            return False
        item.subpixel_grid = cached
        return True

    def _overlay_record_keys(self) -> set[str]:
        keys = set(self._overview_overlay_keys())
        if self._selected_subpixel_selection is not None:
            keys.add(str(self._selected_subpixel_selection.record.key))
        if self._hovered_subpixel_selection is not None:
            keys.add(str(self._hovered_subpixel_selection.record.key))
        return {key for key in keys if key}

    def _create_matrix_item(self, record: FrameRecord, row: int, column: int, index: int) -> _MatrixCellItem:
        x = self._scene_padding + int(column) * (self._cell_size + self._gap)
        y = self._scene_padding + int(row) * (self._cell_size + self._gap)
        item = _MatrixCellItem(QRectF(x, y, self._cell_size, self._cell_size), record, row, column, index)
        item.subpixel_spec = self._subpixel_spec
        item.subpixel_overlay_enabled = self._subpixel_spec is not None
        item.subpixel_grid_provider = self._subpixel_grid_for_record if self._subpixel_spec is not None else None
        grid = self._subpixel_cache_get(self._subpixel_cache_key_for_record(record, self._subpixel_spec)) if self._subpixel_spec is not None else None
        if grid is None:
            grid = getattr(record, "subpixel_grid", None)
        if not isinstance(grid, SubpixelGrid) or self._subpixel_spec is None:
            grid = None
        elif self._subpixel_spec.mode == "tile":
            if (
                grid.spec.mode != "tile"
                or int(grid.spec.tile_width) != int(self._subpixel_spec.tile_width)
                or int(grid.spec.tile_height) != int(self._subpixel_spec.tile_height)
                or int(grid.spec.overlap) != int(self._subpixel_spec.overlap)
            ):
                grid = None
        elif grid.spec.normalized() != self._subpixel_spec:
            grid = None
        item.subpixel_grid = grid
        item.subpixel_color_fn = self._subpixel_color_for_value
        item.subpixel_metric_key = self._metric_key
        item.setToolTip(self._tooltip_for_record(record))
        self._scene.addItem(item)
        self._item_by_key[str(record.key)] = item
        self._apply_item_style(item, sync_tile_state=False)
        self._sync_tile_state_for_keys({record.key})
        return item

    def _ensure_item_for_key(self, key: str | None) -> _MatrixCellItem | None:
        if key is None:
            return None
        normalized_key = str(key)
        item = self._item_by_key.get(normalized_key)
        if item is not None:
            return item
        position = self._record_positions.get(normalized_key)
        record = None if position is None else self._record_by_position.get(position)
        if record is None or position is None:
            return None
        return self._create_matrix_item(record, position[0], position[1], self._record_index_by_key.get(normalized_key, 0))

    def _remove_matrix_item(self, key: str) -> None:
        item = self._item_by_key.pop(str(key), None)
        if item is None:
            return
        if item is self._selected_item:
            self._selected_item = None
        if item is self._hovered_item:
            self._hovered_item = None
        self._scene.removeItem(item)

    def _keys_to_materialize(self) -> set[str]:
        if not self._virtualized_items_enabled:
            return set(self._record_positions)
        visible_keys = self._visible_record_keys(margin_cells=MATRIX_ITEM_KEEP_MARGIN_CELLS)
        max_items = max(1, int(MATRIX_MAX_MATERIALIZED_ITEMS))
        keys = set(visible_keys[:max_items])
        keys.update(self._overlay_record_keys())
        return keys

    def _sync_visible_matrix_items(self, *, force: bool = False) -> None:
        if not self._virtualized_items_enabled:
            return
        keep_keys = self._keys_to_materialize()
        for key in sorted(keep_keys, key=natural_sort_key):
            self._ensure_item_for_key(key)
        overlay_keys = self._overlay_record_keys()
        for key in list(self._item_by_key):
            if key not in keep_keys and key not in overlay_keys:
                self._remove_matrix_item(key)
        self._sync_overview_layer_visibility(force=True)

    def _prepare_visible_tile_queue(self) -> None:
        if self._subpixel_spec is None or not self._tile_overlay_visible:
            return
        generation = self._tile_request_generation
        pending: list[str] = []
        pending_set: set[str] = set()
        for key in self._visible_record_keys(margin_cells=TILE_PREFETCH_MARGIN_CELLS):
            item = self._item_by_key.get(key)
            if item is None:
                continue
            if item.subpixel_grid is not None or self._assign_cached_subpixel_grid(item):
                continue
            pending.append(key)
            pending_set.add(key)
        self._pending_tile_keys = pending
        self._pending_tile_key_set = pending_set
        self._tile_load_generation = generation
        if self._pending_tile_keys:
            self._tile_load_timer.start(0)

    def _process_pending_tile_queue(self) -> None:
        if (
            self._subpixel_spec is None
            or not self._tile_overlay_visible
            or self._tile_load_generation != self._tile_request_generation
        ):
            self._pending_tile_keys.clear()
            self._pending_tile_key_set.clear()
            return
        visible_keys = set(self._visible_record_keys(margin_cells=TILE_PREFETCH_MARGIN_CELLS))
        started = time.perf_counter()
        processed = 0
        while self._pending_tile_keys and processed < TILE_LOAD_MAX_PER_SLICE:
            if (time.perf_counter() - started) * 1000.0 >= TILE_LOAD_SLICE_BUDGET_MS:
                break
            key = self._pending_tile_keys.pop(0)
            self._pending_tile_key_set.discard(key)
            if key not in visible_keys:
                continue
            item = self._item_by_key.get(key)
            if item is None or item.subpixel_grid is not None or self._assign_cached_subpixel_grid(item):
                continue
            grid = self._subpixel_grid_for_record(item.record)
            if self._tile_load_generation != self._tile_request_generation:
                return
            if grid is None:
                continue
            item.subpixel_grid = grid
            item.update()
            processed += 1
        if self._pending_tile_keys:
            self._tile_load_timer.start(0)

    def _sort_records(self, records: list[FrameRecord], sort_mode: str) -> list[FrameRecord]:
        mode = str(sort_mode or "name")
        if mode == "score_desc":
            return sorted(records, key=lambda item: float(item.score), reverse=True)
        if mode == "score_asc":
            return sorted(records, key=lambda item: float(item.score))
        if mode == "input_order":
            return list(records)
        return sorted(records, key=lambda item: natural_sort_key(item.display_name or item.key))

    def _handle_zoom_wheel(self, event) -> bool:
        modifiers = event.modifiers() | QApplication.keyboardModifiers()
        if not bool(modifiers & Qt.KeyboardModifier.ControlModifier):
            return False
        delta_y = event.angleDelta().y() or event.pixelDelta().y()
        if delta_y == 0:
            event.accept()
            return True
        factor = 1.15 if delta_y > 0 else (1.0 / 1.15)
        next_scale = self.transform().m11() * factor
        if MATRIX_MIN_SCALE <= next_scale <= MATRIX_MAX_SCALE:
            self.scale(factor, factor)
            self._emit_overview_state()
        event.accept()
        return True

    def wheelEvent(self, event) -> None:
        if self._handle_zoom_wheel(event):
            self._update_tile_lod()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self._item_for_view_pos(event.pos())
            if item is not None:
                selection = self._tile_selection_for_cell(item, event.pos(), allow_build=True)
                if selection is not None:
                    self._select_tile_selection(selection)
                    self.recordSelected.emit(item.record)
                    self.tileSelected.emit(selection)
                else:
                    self._clear_tile_selection()
                    self._select_item(item)
                    self.recordSelected.emit(item.record)
            else:
                self._clear_tile_selection()
                self._clear_selection()
                self.recordSelected.emit(None)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            item = self._item_for_view_pos(event.pos())
            if item is not None:
                selection = self._tile_selection_for_cell(item, event.pos(), allow_build=True)
                if selection is not None:
                    self._select_tile_selection(selection)
                    self.recordSelected.emit(item.record)
                    self.tileActivated.emit(selection)
                else:
                    self._clear_tile_selection()
                    self._select_item(item)
                    self.recordSelected.emit(item.record)
                    self.recordActivated.emit(item.record)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        item = self._item_for_view_pos(event.pos())
        record = None
        if item is not None:
            selection = self._tile_selection_for_cell(item, event.pos(), allow_build=True)
            if selection is not None:
                self._select_tile_selection(selection)
                self.recordSelected.emit(item.record)
                self.tileSelected.emit(selection)
            else:
                self._clear_tile_selection()
                self._select_item(item)
                self.recordSelected.emit(item.record)
            record = item.record
        self.contextMenuRequested.emit(record, event.globalPos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._pan_active and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._emit_overview_state()
            event.accept()
            return
        item = self._item_for_view_pos(event.pos())
        if item is not None:
            selection = self._tile_selection_for_cell(item, event.pos())
            if selection is not None:
                self._set_hover_tile_selection(selection)
                self._set_hover_item(item)
                QToolTip.showText(event.globalPosition().toPoint(), self._tile_hover_text(selection), self.viewport())
            else:
                self._set_hover_tile_selection(None)
                self._set_hover_item(item)
                QToolTip.showText(event.globalPosition().toPoint(), self._hover_text(item.record), self.viewport())
        else:
            self._set_hover_tile_selection(None)
            self._set_hover_item(None)
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_active:
            self._pan_active = False
            self._pan_start = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_hover_item(None)
        QToolTip.hideText()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._emit_overview_state()
        self._update_tile_lod()

    def _rebuild_scene(self, records: list[FrameRecord]) -> None:
        self._invalidate_tile_requests()
        self._records = list(records)
        ready_scores = [float(record.score) for record in self._records if bool(getattr(record, "score_ready", False))]
        self._auto_color_window_low, self._auto_color_window_high = compute_auto_color_window(ready_scores)
        selected_key = self._selected_item.record.key if self._selected_item is not None else None
        hovered_key = self._hovered_item.record.key if self._hovered_item is not None else None
        self._selected_item = None
        self._hovered_item = None
        self._selected_subpixel_selection = None
        self._hovered_subpixel_selection = None
        self._item_by_key.clear()
        self._record_by_position.clear()
        self._record_positions.clear()
        self._record_index_by_key.clear()
        self._scene.clear()
        self._overview_layer_item = None
        self._matrix_frame_item = None
        self._overview_layer_visible = False
        self._virtualized_items_enabled = False
        if not self._records:
            self._columns = 0
            self._rows = 0
            self._overview_image = None
            self.overviewChanged.emit(None, QRectF(), None, False, tuple(), None)
            return
        placements, self._columns, self._rows = build_matrix_layout(self._records, self._layout_config)
        matrix_rect = self._matrix_scene_rect()
        matrix_width = float(matrix_rect.width())
        matrix_height = float(matrix_rect.height())
        self._scene.setSceneRect(0, 0, matrix_width + self._scene_padding * 2, matrix_height + self._scene_padding * 2)
        self._matrix_frame_item = self._scene.addRect(matrix_rect, QPen(SUBDUED_TEXT_COLOR, 1.0))
        self._matrix_frame_item.setZValue(-5.0)
        self._virtualized_items_enabled = len(placements) >= MATRIX_VIRTUALIZE_RECORD_THRESHOLD
        for index, (record, row, column) in enumerate(placements):
            self._record_positions[record.key] = (row, column)
            self._record_by_position[(row, column)] = record
            self._record_index_by_key[record.key] = index
            if not self._virtualized_items_enabled:
                item = self._create_matrix_item(record, row, column, index)
                if selected_key == record.key:
                    self._selected_item = item
                if hovered_key == record.key:
                    self._hovered_item = item
        if self._virtualized_items_enabled:
            self._selected_item = self._ensure_item_for_key(selected_key)
            self._hovered_item = self._ensure_item_for_key(hovered_key)
        if self._selected_item is not None:
            self._apply_item_style(self._selected_item, sync_tile_state=False)
        if self._hovered_item is not None:
            self._apply_item_style(self._hovered_item, sync_tile_state=False)
        if not self._virtualized_items_enabled:
            self._sync_tile_state_for_keys(self._item_by_key.keys())
        self._overview_image = self._build_overview_image(placements)
        self._refresh_overview_layer_pixmap()
        self._sync_visible_matrix_items(force=True)
        self._emit_overview_state()
        self._update_tile_lod(force=True)

    def _build_overview_image(self, placements: list[tuple[FrameRecord, int, int]]) -> QImage:
        image = QImage(max(1, self._columns), max(1, self._rows), QImage.Format.Format_RGB888)
        image.fill(MATRIX_BACKGROUND_ALT)
        for record, row, column in placements:
            image.setPixelColor(column, row, self._background_color_for_record(record))
        return image

    def _matrix_scene_rect(self) -> QRectF:
        matrix_width = self._columns * (self._cell_size + self._gap)
        matrix_height = self._rows * (self._cell_size + self._gap)
        return QRectF(self._scene_padding, self._scene_padding, matrix_width, matrix_height)

    def _refresh_overview_layer_pixmap(self) -> None:
        if self._overview_image is None or self._overview_image.isNull() or self._columns <= 0 or self._rows <= 0:
            self._overview_layer_item = None
            return
        matrix_rect = self._matrix_scene_rect()
        pixmap = QPixmap.fromImage(self._overview_image)
        if self._overview_layer_item is None:
            item = QGraphicsPixmapItem()
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            item.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
            item.setZValue(-10.0)
            self._scene.addItem(item)
            self._overview_layer_item = item
        self._overview_layer_item.setPixmap(pixmap)
        self._overview_layer_item.setPos(matrix_rect.left(), matrix_rect.top())
        self._overview_layer_item.setTransform(
            QTransform.fromScale(
                matrix_rect.width() / max(1.0, float(pixmap.width())),
                matrix_rect.height() / max(1.0, float(pixmap.height())),
            )
        )
        self._overview_layer_item.setVisible(self._overview_layer_visible)

    def _overview_overlay_keys(self) -> set[str]:
        keys: set[str] = set(self._processing_keys)
        if self._selected_item is not None:
            keys.add(str(self._selected_item.record.key))
        if self._hovered_item is not None:
            keys.add(str(self._hovered_item.record.key))
        if self._reference_key:
            keys.add(str(self._reference_key))
        return keys

    def _overview_layer_should_be_active(self) -> bool:
        if self._overview_image is None or self._overview_image.isNull():
            return False
        if len(self._records) < LOW_ZOOM_OVERVIEW_RECORD_THRESHOLD:
            return False
        zoom_level = max(0.01, abs(float(self.transform().m11())))
        return zoom_level <= LOW_ZOOM_OVERVIEW_MAX_ZOOM

    def _sync_overview_layer_visibility(self, *, force: bool = False) -> bool:
        use_overview = self._overview_layer_should_be_active()
        previous = self._overview_layer_visible
        self._overview_layer_visible = use_overview
        if self._overview_layer_item is not None and (force or use_overview != previous or self._overview_layer_item.isVisible() != use_overview):
            self._overview_layer_item.setVisible(use_overview)
        if force or use_overview != previous:
            overlay_keys = self._overview_overlay_keys() if use_overview else set()
            for key, item in self._item_by_key.items():
                item.setVisible((not use_overview) or (key in overlay_keys))
        return use_overview != previous

    def _sync_low_zoom_visibility_for_keys(self, keys) -> None:
        if not self._overview_layer_visible:
            return
        overlay_keys = self._overview_overlay_keys()
        normalized_keys = {str(key) for key in keys if key}
        if not normalized_keys:
            return
        for key in normalized_keys:
            item = self._ensure_item_for_key(key) if key in overlay_keys else self._item_by_key.get(key)
            if item is not None:
                item.setVisible(key in overlay_keys)

    def _item_from_scene_pos(self, scene_pos) -> _MatrixCellItem | None:
        x = float(scene_pos.x()) - float(self._scene_padding)
        y = float(scene_pos.y()) - float(self._scene_padding)
        if x < 0.0 or y < 0.0:
            return None
        span = float(self._cell_size + self._gap)
        if span <= 0.0:
            return None
        column = int(x // span)
        row = int(y // span)
        if row < 0 or column < 0 or row >= self._rows or column >= self._columns:
            return None
        local_x = x - float(column) * span
        local_y = y - float(row) * span
        if local_x > float(self._cell_size) or local_y > float(self._cell_size):
            return None
        record = self._record_by_position.get((row, column))
        if record is None:
            return None
        return self._ensure_item_for_key(str(record.key))

    def _item_for_view_pos(self, view_pos) -> _MatrixCellItem | None:
        item = self.itemAt(view_pos)
        if isinstance(item, _MatrixCellItem):
            return item
        return self._item_from_scene_pos(self.mapToScene(view_pos))

    def _emit_overview_state(self) -> None:
        if self._overview_image is None or self._columns <= 0 or self._rows <= 0:
            self.overviewChanged.emit(None, QRectF(), None, False, tuple(), None)
            return
        scene_rect = self._scene.sceneRect()
        visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        selected_position = None if self._selected_item is None else (self._selected_item.row, self._selected_item.column)
        processing_positions = tuple(self._record_positions[key] for key in self._processing_keys if key in self._record_positions)
        reference_position = self._record_positions.get(self._reference_key) if self._reference_key in self._record_positions else None
        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            self.overviewChanged.emit(self._overview_image, QRectF(), selected_position, self._selection_blink_on, processing_positions, reference_position)
            return
        left_padding = self._scene_padding / max(1.0, scene_rect.width())
        top_padding = self._scene_padding / max(1.0, scene_rect.height())
        width_padding = (self._scene_padding * 2) / max(1.0, scene_rect.width())
        height_padding = (self._scene_padding * 2) / max(1.0, scene_rect.height())
        normalized = QRectF(
            max(0.0, (visible_scene_rect.left() / scene_rect.width() - left_padding) / max(NORMALIZATION_EPSILON, 1.0 - width_padding)),
            max(0.0, (visible_scene_rect.top() / scene_rect.height() - top_padding) / max(NORMALIZATION_EPSILON, 1.0 - height_padding)),
            min(1.0, (visible_scene_rect.width() / scene_rect.width()) / max(NORMALIZATION_EPSILON, 1.0 - width_padding)),
            min(1.0, (visible_scene_rect.height() / scene_rect.height()) / max(NORMALIZATION_EPSILON, 1.0 - height_padding)),
        )
        self.overviewChanged.emit(self._overview_image, normalized, selected_position, self._selection_blink_on, processing_positions, reference_position)

    def _select_item(self, item: _MatrixCellItem) -> None:
        if self._selected_item is item:
            return
        previous = self._selected_item
        self._selected_item = item
        self._selection_blink_on = True
        if previous is not None:
            self._apply_item_style(previous)
        self._apply_item_style(item)
        self._sync_low_zoom_visibility_for_keys({None if previous is None else previous.record.key, item.record.key})
        self._emit_overview_state()

    def _clear_tile_selection(self) -> None:
        if self._selected_subpixel_selection is not None:
            previous = self._selected_subpixel_selection
            self._selected_subpixel_selection = None
            self._sync_tile_state_for_keys({previous.record.key})
        if self._hovered_subpixel_selection is not None:
            previous_hover = self._hovered_subpixel_selection
            self._hovered_subpixel_selection = None
            self._sync_tile_state_for_keys({previous_hover.record.key})

    def _select_tile_selection(self, selection: MatrixTileSelection) -> None:
        previous = self._selected_subpixel_selection
        self._selected_subpixel_selection = selection
        keys = {selection.record.key}
        if previous is not None:
            keys.add(previous.record.key)
        self._select_item(self._item_by_key[selection.record.key])
        self._sync_tile_state_for_keys(keys)
        self._emit_overview_state()
        self.tileSelected.emit(selection)

    def _set_hover_tile_selection(self, selection: MatrixTileSelection | None) -> None:
        previous = self._hovered_subpixel_selection
        if previous is selection:
            return
        self._hovered_subpixel_selection = selection
        keys = set()
        if previous is not None:
            keys.add(previous.record.key)
        if selection is not None:
            keys.add(selection.record.key)
        if keys:
            self._sync_tile_state_for_keys(keys)

    def _tile_hover_text(self, selection: MatrixTileSelection) -> str:
        payload = selection
        return (
            f"{payload.record.display_name} | parent r{payload.matrix_row + 1}, c{payload.matrix_column + 1}"
            f" | subpixel r{payload.sub_row + 1}, c{payload.sub_column + 1}"
            f" | value {payload.subpixel_value:.4f}"
            f" | parent {payload.parent_value:.4f}"
        )

    def _subpixel_overlay_visible(self, spec: SubpixelGridSpec | None = None) -> bool:
        active_spec = self._subpixel_spec if spec is None else spec
        if active_spec is None:
            return False
        zoom_level = max(0.01, abs(float(self.transform().m11())))
        probe_rect = QRectF(0.0, 0.0, float(self._cell_size), float(self._cell_size))
        return _subpixel_overlay_visible_for_rect(
            probe_rect,
            active_spec,
            zoom_level,
            zoom_threshold=self._tile_zoom_threshold,
        )

    def _update_tile_lod(self, *, force: bool = False) -> None:
        show_tiles = self._subpixel_overlay_visible()
        previous_visibility = self._tile_overlay_visible
        previous_lod_band = self._active_lod_band
        self._tile_overlay_visible = show_tiles
        if not show_tiles:
            self._set_hover_tile_selection(None)
        overview_changed = self._sync_overview_layer_visibility(force=force)
        self._active_lod_band = self._lod_band_for_current_view()
        lod_changed = self._active_lod_band != previous_lod_band
        if force or show_tiles != previous_visibility or overview_changed or lod_changed:
            self.viewport().update()
        self._sync_visible_matrix_items()
        if show_tiles:
            self._schedule_visible_tile_request(immediate=force or show_tiles != previous_visibility or lod_changed)
        else:
            self._invalidate_tile_requests()

    def _clear_selection(self) -> None:
        if self._selected_item is not None:
            previous = self._selected_item
            self._selected_item = None
            self._selection_blink_on = False
            self._apply_item_style(previous)
            self._sync_low_zoom_visibility_for_keys({previous.record.key})
        self._emit_overview_state()

    def _set_hover_item(self, item: _MatrixCellItem | None) -> None:
        if self._hovered_item is item:
            return
        previous = self._hovered_item
        self._hovered_item = item
        if previous is not None:
            self._apply_item_style(previous)
        if item is not None:
            self._apply_item_style(item)
        self._sync_low_zoom_visibility_for_keys(
            {
                None if previous is None else previous.record.key,
                None if item is None else item.record.key,
            }
        )

    def _apply_item_style(self, item: _MatrixCellItem, *, sync_tile_state: bool = True) -> None:
        if item is self._hovered_item:
            pen = QPen(HOVER_BORDER, MATRIX_HOVER_PEN_WIDTH)
        elif item.record.key in self._processing_keys:
            pen = QPen(PROCESSING_BORDER, MATRIX_PROCESSING_PEN_WIDTH)
        elif self._reference_key is not None and item.record.key == self._reference_key:
            pen = QPen(REFERENCE_BORDER, MATRIX_REFERENCE_PEN_WIDTH)
        else:
            pen = QPen(DEFAULT_BORDER, MATRIX_DEFAULT_PEN_WIDTH)
        brush_color = self._background_color_for_record(item.record)
        if item is self._selected_item and self._selection_blink_on:
            brush_color = blend_colors(brush_color, SELECTED_BLINK_COLOR, MATRIX_SELECTED_BLEND_RATIO)
        item.setPen(pen)
        item.setBrush(brush_color)
        if sync_tile_state:
            self._sync_tile_state_for_keys({item.record.key})

    def _sync_tile_state_for_keys(self, record_keys) -> None:
        keys = {str(key) for key in record_keys}
        if not keys:
            return
        for key in keys:
            item = self._item_by_key.get(key)
            if item is None:
                continue
            selected_tile = self._selected_subpixel_selection if self._selected_subpixel_selection is not None and self._selected_subpixel_selection.record.key == key else None
            hovered_tile = self._hovered_subpixel_selection if self._hovered_subpixel_selection is not None and self._hovered_subpixel_selection.record.key == key else None
            item.set_tile_state(selected_tile, hovered_tile)

    def _tile_overlay_active(self) -> bool:
        return self._tile_overlay_visible

    def _tile_selection_for_cell(self, item: _MatrixCellItem, view_pos, *, allow_build: bool = False) -> MatrixTileSelection | None:
        spec = self._subpixel_spec
        if spec is None or not self._tile_overlay_active():
            return None
        rect = item.rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return None
        scene_pos = self.mapToScene(view_pos)
        local_pos = item.mapFromScene(scene_pos)
        local_x = float(local_pos.x()) - float(rect.left())
        local_y = float(local_pos.y()) - float(rect.top())
        if local_x < 0.0 or local_y < 0.0 or local_x > rect.width() or local_y > rect.height():
            return None
        subpixel_grid = item.subpixel_grid
        if subpixel_grid is None and self._assign_cached_subpixel_grid(item):
            subpixel_grid = item.subpixel_grid
        if subpixel_grid is None and allow_build:
            subpixel_grid = self._subpixel_grid_for_record(item.record)
            item.subpixel_grid = subpixel_grid
        if subpixel_grid is None:
            if str(item.record.key) not in self._pending_tile_key_set:
                self._schedule_visible_tile_request()
            return None
        actual_spec = subpixel_grid.spec if subpixel_grid is not None else spec
        resolved = _display_tile_index_for_cell(local_x, local_y, rect, actual_spec)
        if resolved is None:
            return None
        tile_row, tile_column = resolved
        if subpixel_grid is not None:
            value = subpixel_grid.value_at(tile_row, tile_column)
            confidence = subpixel_grid.confidence_at(tile_row, tile_column)
            parent_value = subpixel_grid.aggregate_value(self._subpixel_aggregation)
        else:
            value = float(item.record.score if bool(getattr(item.record, "score_ready", False)) else 0.0)
            confidence = None
            parent_value = value
        return MatrixTileSelection(
            record=item.record,
            matrix_row=item.row,
            matrix_column=item.column,
            sub_row=tile_row,
            sub_column=tile_column,
            spec=actual_spec,
            parent_value=float(parent_value),
            subpixel_value=float(value),
            subpixel_confidence=confidence,
            aggregation=self._subpixel_aggregation,
            metric_key=str(self._metric_key or "overall_frame_score"),
        )

    def _toggle_selection_blink(self) -> None:
        if self._selected_item is None:
            if self._selection_blink_on:
                self._selection_blink_on = False
                self._emit_overview_state()
            return
        self._selection_blink_on = not self._selection_blink_on
        self._apply_item_style(self._selected_item)
        self._emit_overview_state()

    def _display_score(self, record: FrameRecord) -> float | None:
        if not bool(getattr(record, "score_ready", False)):
            return None
        if self._score_view_mode == "absolute":
            absolute_value = getattr(record, "absolute_score", None)
            if absolute_value is None:
                return None
            ratio = metric_visual_ratio(
                self._metric_key,
                float(absolute_value),
                point_match_radius=float(self._point_match_radius),
                bce_score_cap=float(self._bce_score_cap),
            )
            if ratio is None:
                return None
            higher_is_better = metric_higher_is_better(str(self._metric_key or ""))
            goodness = float(ratio) if higher_is_better else (1.0 - float(ratio))
            return max(0.0, min(goodness, 1.0))
        return float(record.score)

    @staticmethod
    def _format_metric_value(value: float) -> str:
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return f"{numeric * 100.0:.3f}%"
        return f"{numeric:.3f}"

    def _background_color(self, score: float) -> QColor:
        if self._score_view_mode == "absolute":
            position = max(0.0, min(float(score), 1.0))
        else:
            position = map_score_to_palette_position(score, self._auto_color_window_low, self._auto_color_window_high)
        position = enhance_palette_position(position)
        return interpolate_gradient_color(self._gradient_name, position)

    def _subpixel_color_for_value(self, score: float) -> QColor:
        metric_key = str(self._metric_key or "")
        ratio = metric_visual_ratio(
            metric_key,
            float(score),
            point_match_radius=float(self._point_match_radius),
            bce_score_cap=float(self._bce_score_cap),
        )
        if ratio is None:
            return QColor(MATRIX_BACKGROUND_ALT)
        level_key = metric_level_key(
            metric_key,
            float(score),
            point_match_radius=float(self._point_match_radius),
            bce_score_cap=float(self._bce_score_cap),
        )
        family = metric_key.split("::", 1)[0]
        higher_is_better = metric_higher_is_better(metric_key)
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

    def _background_color_for_record(self, record: FrameRecord) -> QColor:
        score = self._display_score(record)
        if score is None:
            return QColor(MATRIX_BACKGROUND_ALT)
        return self._background_color(score)

    def _tooltip_for_record(self, record: FrameRecord) -> str:
        if not bool(getattr(record, "score_ready", False)):
            suffix = f"\n{self._t('matrix.reference_frame')}" if self._reference_key == record.key else ""
            return f"{record.display_name}\n{self._t('matrix.mismatch_not_computed')}{suffix}"
        lines = [record.display_name]
        if record.absolute_score is not None:
            lines.append(f"{self._t('matrix.absolute_mismatch')}: {self._format_metric_value(record.absolute_score)}")
        if record.relative_score is not None:
            lines.append(f"{self._t('matrix.relative_mismatch')}: {record.relative_score * 100.0:.3f}%")
        if record.score_percentile is not None:
            lines.append(f"Score percentile: P{float(record.score_percentile):.1f}")
        if self._reference_key == record.key:
            lines.append(self._t('matrix.reference_frame'))
        return "\n".join(lines)

    def _hover_text(self, record: FrameRecord) -> str:
        if not bool(getattr(record, "score_ready", False)):
            return f"{record.display_name} | {self._t('matrix.mismatch_not_computed').lower()}"
        parts = [record.display_name]
        if self._reference_key == record.key:
            parts.append(self._t('matrix.reference_short'))
        if record.absolute_score is not None:
            parts.append(f"{self._t('matrix.absolute_short')} {self._format_metric_value(record.absolute_score)}")
        if record.relative_score is not None:
            parts.append(f"{self._t('matrix.relative_short')} {record.relative_score * 100.0:.3f}%")
        if record.score_percentile is not None:
            parts.append(f"P{float(record.score_percentile):.1f}")
        return " | ".join(parts)


__all__ = [
    "GradientPresetSelectorWidget",
    "GradientRangeSelectorWidget",
    "compute_auto_color_window",
    "enhance_palette_position",
    "MatrixLayoutConfig",
    "MatrixTileSelection",
    "MatrixListWidget",
    "MatrixMiniMapWidget",
    "blend_colors",
    "build_matrix_layout",
    "error_palette_color",
    "extract_frame_number",
    "interpolate_gradient_color",
    "map_score_to_palette_position",
]





