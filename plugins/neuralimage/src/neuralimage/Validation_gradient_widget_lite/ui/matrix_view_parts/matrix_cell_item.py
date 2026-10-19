"""matrix_view: matrix cell item."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsRectItem,
)
from ...core.domain import FrameRecord
from ...core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
)
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    DEFAULT_GRADIENT_NAME,
    MATRIX_BACKGROUND_ALT,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import MIN_VISIBLE_TILE_SCREEN_SIZE

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.compute_auto_color_window import interpolate_gradient_color

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.score_fill_color import (
    _subpixel_overlay_visible_for_rect,
    _tile_rect_for_cell,
    _tile_screen_extent_for_rect,
)

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

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.score_fill_color import MatrixTileSelection
