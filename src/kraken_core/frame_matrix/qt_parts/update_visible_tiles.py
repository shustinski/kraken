"""qt: update visible tiles."""
from __future__ import annotations
from typing import TYPE_CHECKING
import math
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter

from kraken_core.frame_matrix.mosaic import MOSAIC_EDGE
from kraken_core.frame_matrix.qt_parts.module_state import (
    GridOrientation,
)

from kraken_core.frame_matrix.qt_parts.frame_rect import FrameRect

from kraken_core.frame_matrix.qt_parts.matrix_lod import MatrixLod

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import (
    FrameSelection,
    _ViewportTileItem,
)


class UpdateVisibleTilesMixin:
    def center_on_frame(self, x: int, y: int) -> None:
        self.centerOn(self.scene_rect_for_frame(x, y).center())
        self._schedule_visible_update()

    def zoom_factor(self) -> float:
        return abs(float(self.transform().m11()))

    def _fitted_zoom_factor(self) -> float:
        """Largest scale at which the whole layer still covers the viewport."""

        scene_rect = self.sceneRect()
        viewport = self.viewport().rect()
        if (
            scene_rect.width() <= 0
            or scene_rect.height() <= 0
            or viewport.width() <= 0
            or viewport.height() <= 0
        ):
            return self.MIN_ZOOM
        width = viewport.width()
        height = viewport.height()
        vertical_bar = self.verticalScrollBar()
        horizontal_bar = self.horizontalScrollBar()
        if vertical_bar.isVisible():
            width += vertical_bar.width()
        if horizontal_bar.isVisible():
            height += horizontal_bar.height()
        fitted = min(width / scene_rect.width(), height / scene_rect.height())
        return max(self.MIN_ZOOM, min(self.MAX_ZOOM, fitted))

    def lod_level(self) -> MatrixLod:
        zoom = self.zoom_factor()
        if zoom < self.CELLS_LOD:
            return MatrixLod.OVERVIEW
        if zoom < self.DETAILS_LOD:
            return MatrixLod.CELLS
        return MatrixLod.DETAILS

    def reset_zoom(self) -> None:
        self._stop_zoom_animation(rebuild=False)
        self.resetTransform()
        self._after_transform()

    def set_zoom_factor(self, factor: float) -> None:
        self._stop_zoom_animation(rebuild=False)
        bounded = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(factor)))
        self.resetTransform()
        self.scale(bounded, bounded)
        self._after_transform()

    def zoom_to_fit(self) -> None:
        self._stop_zoom_animation(rebuild=False)
        scene_rect = self.sceneRect()
        if scene_rect.isEmpty():
            return
        self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
        if self.zoom_factor() < self.MIN_ZOOM:
            self.set_zoom_factor(self.MIN_ZOOM)
            return
        self._after_transform()

    def visible_item_count(self) -> int:
        return len(self._visible_tiles)

    def visible_tile_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._visible_tiles))

    def visible_scene_rect(self) -> QRectF:
        return self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())

    def _ensure_inside(self, x: int, y: int) -> None:
        if not 1 <= int(x) <= self.matrix_width or not 1 <= int(y) <= self.matrix_height:
            raise ValueError(
                f"frame coordinate ({x}, {y}) is outside {self.matrix_width} x {self.matrix_height} matrix"
            )

    def _row_for_y(self, y: int) -> int:
        if self._orientation is GridOrientation.Y_DOWN:
            return y - 1
        return self.matrix_height - y

    def _y_for_row(self, row: int) -> int:
        if self._orientation is GridOrientation.Y_DOWN:
            return row + 1
        return self.matrix_height - row

    def _clamp_selection(self, selection: FrameSelection) -> FrameSelection:
        rectangles: list[FrameRect] = []
        for rectangle in selection.rectangles:
            if rectangle.x1 > self.matrix_width or rectangle.y1 > self.matrix_height:
                continue
            rectangles.append(
                FrameRect(
                    rectangle.x1,
                    rectangle.y1,
                    min(rectangle.x2, self.matrix_width),
                    min(rectangle.y2, self.matrix_height),
                )
            )
        return FrameSelection(tuple(rectangles))

    def _effective_tile_cells(self) -> int:
        zoom = max(self.MIN_ZOOM, self.zoom_factor())
        required = max(self.BASE_TILE_CELLS, math.ceil(self.TILE_DEVICE_TARGET / (self.CELL_PITCH * zoom)))
        return 1 << (required - 1).bit_length()

    def _mosaic_tile_cells(self) -> int:
        """Mosaic level whose per-cell slot is at least as large as a cell on screen.

        Graphics tiles never drop below ``BASE_TILE_CELLS``, but at close zoom a
        16-cell mosaic gives every frame only a few pixels. Finer mosaics cover
        the same graphics tile in pieces, so the image stays sharp.
        """

        view_cells = self._effective_tile_cells()
        try:
            ratio = float(self.viewport().devicePixelRatioF())
        except (AttributeError, RuntimeError):
            ratio = 1.0
        device_cell = self.CELL_PITCH * max(self.MIN_ZOOM, self.zoom_factor()) * max(ratio, 1.0)
        fit = MOSAIC_EDGE / max(device_cell, 1.0e-9)
        cells = self.FINEST_MOSAIC_CELLS
        while cells * 2 <= fit and cells * 2 <= view_cells:
            cells *= 2
        return min(view_cells, cells)

    def _rebuild_tile_index(self, tile_cells: int) -> None:
        buckets: dict[tuple[int, int], list[FrameCellData]] = {}
        for cell in self._cells.values():
            column = cell.x - 1
            row = self._row_for_y(cell.y)
            buckets.setdefault((column // tile_cells, row // tile_cells), []).append(cell)
        self._tile_index = {key: tuple(value) for key, value in buckets.items()}
        self._indexed_tile_cells = tile_cells

    def _schedule_visible_update(self, *_args: object) -> None:
        if self._update_scheduled:
            return
        self._update_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_update)

    def _run_scheduled_update(self) -> None:
        self._update_scheduled = False
        if getattr(self, "_zoom_animating", False):
            return
        self._update_visible_tiles()

    def _update_visible_tiles(self) -> None:
        if self.viewport().width() <= 0 or self.viewport().height() <= 0:
            return
        tile_cells = self._effective_tile_cells()
        if tile_cells != self._indexed_tile_cells:
            self._rebuild_tile_index(tile_cells)
            self._clear_tiles()

        visible = self.visible_scene_rect()
        if visible.isEmpty():
            return
        tile_span = tile_cells * self.CELL_PITCH
        max_tile_x = (self.matrix_width - 1) // tile_cells
        max_tile_y = (self.matrix_height - 1) // tile_cells
        first_x = max(0, int(math.floor(visible.left() / tile_span)) - 1)
        last_x = min(max_tile_x, int(math.floor(visible.right() / tile_span)) + 1)
        first_y = max(0, int(math.floor(visible.top() / tile_span)) - 1)
        last_y = min(max_tile_y, int(math.floor(visible.bottom() / tile_span)) + 1)
        desired = {
            (tile_x, tile_y)
            for tile_y in range(first_y, last_y + 1)
            for tile_x in range(first_x, last_x + 1)
        }
        for key in tuple(self._visible_tiles):
            if key in desired:
                continue
            item = self._visible_tiles.pop(key)
            self._graphics_scene.removeItem(item)
        for key in desired - self._visible_tiles.keys():
            item = _ViewportTileItem(self, key[0], key[1], tile_cells)
            self._visible_tiles[key] = item
            self._graphics_scene.addItem(item)

        lod = self.lod_level()
        if lod is not self._last_lod:
            self._last_lod = lod
            self.lodChanged.emit(lod.value)
        self.viewportChanged.emit(visible)

    def _clear_tiles(self) -> None:
        for item in self._visible_tiles.values():
            self._graphics_scene.removeItem(item)
        self._visible_tiles.clear()

    def _invalidate_tiles(self) -> None:
        if self._indexed_tile_cells == 0:
            self._update_visible_tiles()
        for item in self._visible_tiles.values():
            item.update()
        overlay = getattr(self, "_selection_overlay", None)
        if overlay is not None:
            overlay.update()
        self.viewport().update()
        for child in self.viewport().children():
            if child.objectName() == "matrixMinimap":
                child.update()

    def _status_color(self, status: str) -> QColor:
        candidate = QColor(self._status_colors.get(status, self._status_colors["empty"]))
        return candidate if candidate.isValid() else QColor("#1f2937")

    def _paint_overview_tile(self, painter: QPainter, tile: _ViewportTileItem) -> None:
        """Far zoom shows the continuous layer image, not one status block per tile."""

        if self._paint_loaded_mosaic(painter, tile):
            return
        painter.fillRect(tile.boundingRect(), self._status_color("empty"))

if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData
