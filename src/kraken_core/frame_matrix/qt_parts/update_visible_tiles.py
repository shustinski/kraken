"""qt: update visible tiles."""
from __future__ import annotations
from typing import TYPE_CHECKING
import math
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen

from kraken_core.frame_matrix.qt_parts.module_state import (
    GridOrientation,
    STATUS_PRIORITY,
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

    def lod_level(self) -> MatrixLod:
        zoom = self.zoom_factor()
        if zoom < self.CELLS_LOD:
            return MatrixLod.OVERVIEW
        if zoom < self.DETAILS_LOD:
            return MatrixLod.CELLS
        return MatrixLod.DETAILS

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._after_transform()

    def set_zoom_factor(self, factor: float) -> None:
        bounded = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(factor)))
        self.resetTransform()
        self.scale(bounded, bounded)
        self._after_transform()

    def zoom_to_fit(self) -> None:
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
        self.viewport().update()

    def _status_color(self, status: str) -> QColor:
        candidate = QColor(self._status_colors.get(status, self._status_colors["empty"]))
        return candidate if candidate.isValid() else QColor("#1f2937")

    def _tile_frame_rect(self, tile: _ViewportTileItem) -> FrameRect:
        x1 = tile.tile_x * tile.tile_cells + 1
        x2 = min(self.matrix_width, x1 + tile.tile_cells - 1)
        row1 = tile.tile_y * tile.tile_cells
        row2 = min(self.matrix_height - 1, row1 + tile.tile_cells - 1)
        y_first = self._y_for_row(row1)
        y_last = self._y_for_row(row2)
        return FrameRect(x1, min(y_first, y_last), x2, max(y_first, y_last))

    def _paint_overview_tile(self, painter: QPainter, tile: _ViewportTileItem) -> None:
        materialized = self._tile_index.get((tile.tile_x, tile.tile_y), ())
        highest_status = "empty"
        highest_priority = STATUS_PRIORITY["empty"]
        for cell in materialized:
            priority = STATUS_PRIORITY.get(cell.status, 1)
            if priority > highest_priority:
                highest_status = cell.status
                highest_priority = priority
        tile_frames = self._tile_frame_rect(tile)
        for aggregate in self._aggregates:
            bounds = aggregate.bounds
            if not tile_frames.intersects(FrameRect(bounds.x1, bounds.y1, bounds.x2, bounds.y2)):
                continue
            for status, count in aggregate.status_counts.items():
                if int(count) <= 0:
                    continue
                priority = STATUS_PRIORITY.get(str(status), 1)
                if priority > highest_priority:
                    highest_status = str(status)
                    highest_priority = priority
        rect = tile.boundingRect().adjusted(0.8, 0.8, -1.2, -1.2)
        painter.fillRect(rect, self._status_color(highest_status))
        border = QColor("#475569")
        if any(rectangle.intersects(tile_frames) for rectangle in self._selection.rectangles):
            border = QColor("#f8fafc")
        pen = QPen(border, max(1.0, 1.5 / max(self.zoom_factor(), self.MIN_ZOOM)))
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(rect)

if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData
