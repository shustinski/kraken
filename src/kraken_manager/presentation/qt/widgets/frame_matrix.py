"""Virtualized, domain-neutral frame matrix for project workspaces."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QContextMenuEvent, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)

from .grid_dimensions import GridOrientation


class MatrixLod(StrEnum):
    """Rendering detail exposed for toolbars and diagnostics."""

    OVERVIEW = "overview"
    CELLS = "cells"
    DETAILS = "details"


@dataclass(frozen=True, slots=True)
class FrameRect:
    """Inclusive, one-based rectangle of project coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in (self.x1, self.y1, self.x2, self.y2))
        if min(values) < 1:
            raise ValueError("frame coordinates are one-based and must be positive")
        left, right = sorted((values[0], values[2]))
        top, bottom = sorted((values[1], values[3]))
        object.__setattr__(self, "x1", left)
        object.__setattr__(self, "y1", top)
        object.__setattr__(self, "x2", right)
        object.__setattr__(self, "y2", bottom)

    @classmethod
    def from_points(cls, first: tuple[int, int], second: tuple[int, int]) -> FrameRect:
        return cls(first[0], first[1], second[0], second[1])

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def frame_count(self) -> int:
        return self.width * self.height

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersects(self, other: FrameRect) -> bool:
        return not (
            self.x2 < other.x1
            or other.x2 < self.x1
            or self.y2 < other.y1
            or other.y2 < self.y1
        )

    def coordinates(self) -> Iterator[tuple[int, int]]:
        for y in range(self.y1, self.y2 + 1):
            for x in range(self.x1, self.x2 + 1):
                yield x, y


@dataclass(frozen=True, slots=True)
class FrameSelection:
    """Compact selection represented by rectangles, not millions of IDs."""

    rectangles: tuple[FrameRect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rectangles", tuple(self.rectangles))

    @classmethod
    def single(cls, x: int, y: int) -> FrameSelection:
        return cls((FrameRect(x, y, x, y),))

    @property
    def is_empty(self) -> bool:
        return not self.rectangles

    def contains(self, x: int, y: int) -> bool:
        return any(rectangle.contains(x, y) for rectangle in self.rectangles)

    def coordinates(self, *, maximum: int | None = None) -> Iterator[tuple[int, int]]:
        """Iterate unique coordinates, optionally protecting callers from expansion."""
        seen: set[tuple[int, int]] = set()
        for rectangle in self.rectangles:
            for coordinate in rectangle.coordinates():
                if coordinate in seen:
                    continue
                seen.add(coordinate)
                if maximum is not None and len(seen) > maximum:
                    raise ValueError(f"selection contains more than {maximum} frames")
                yield coordinate


@dataclass(frozen=True, slots=True)
class FrameCellData:
    """Presentation data for one materialized layer/frame state."""

    x: int
    y: int
    status: str = "empty"
    performer_color: str | None = None
    performer_initials: str = ""
    label: str = ""
    tooltip: str = ""
    thumbnail: QPixmap | None = field(default=None, compare=False, repr=False)
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.x) < 1 or int(self.y) < 1:
            raise ValueError("frame coordinates are one-based and must be positive")
        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))
        object.__setattr__(self, "status", str(self.status or "empty"))


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Value emitted for a frame context menu request."""

    x: int
    y: int
    cell: FrameCellData
    selection: FrameSelection


DEFAULT_STATUS_COLORS: dict[str, str] = {
    "empty": "#1f2937",
    "image_ready": "#2563eb",
    "processing": "#7c3aed",
    "vectorized": "#0891b2",
    "in_review": "#d97706",
    "returned_unchanged": "#65a30d",
    "returned_changed": "#ca8a04",
    "approved": "#15803d",
    "changes_requested": "#ea580c",
    "conflict": "#dc2626",
    "error": "#b91c1c",
}

STATUS_PRIORITY = {
    "empty": 0,
    "image_ready": 1,
    "vectorized": 2,
    "processing": 3,
    "in_review": 4,
    "returned_unchanged": 5,
    "returned_changed": 6,
    "approved": 7,
    "changes_requested": 8,
    "conflict": 9,
    "error": 10,
}


class _ViewportTileItem(QGraphicsItem):
    """One visible tile; it paints either an aggregate or its individual cells."""

    def __init__(self, owner: FrameMatrixView, tile_x: int, tile_y: int, tile_cells: int) -> None:
        super().__init__()
        self.owner = owner
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.tile_cells = tile_cells
        pitch = owner.CELL_PITCH
        left = tile_x * tile_cells * pitch
        top = tile_y * tile_cells * pitch
        columns = min(tile_cells, owner.matrix_width - tile_x * tile_cells)
        rows = min(tile_cells, owner.matrix_height - tile_y * tile_cells)
        self._bounds = QRectF(left, top, columns * pitch, rows * pitch)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return self._bounds

    def paint(  # type: ignore[override]
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        del option, widget
        lod = self.owner.lod_level()
        if lod is MatrixLod.OVERVIEW or self.tile_cells > self.owner.BASE_TILE_CELLS:
            self.owner._paint_overview_tile(painter, self)
            return
        self.owner._paint_tile_cells(painter, self, lod)


class FrameMatrixView(QGraphicsView):
    """Sparse frame matrix with tile virtualization, LOD and rectangle selection.

    The view never creates an object for every project frame. It maintains only
    tiles intersecting the viewport; tile size grows as the user zooms out.
    """

    selectionChanged = pyqtSignal(object)
    frameActivated = pyqtSignal(int, int)
    contextMenuRequested = pyqtSignal(object, object)
    viewportChanged = pyqtSignal(object)
    lodChanged = pyqtSignal(str)

    CELL_SIZE = 48.0
    CELL_GAP = 3.0
    CELL_PITCH = CELL_SIZE + CELL_GAP
    BASE_TILE_CELLS = 16
    TILE_DEVICE_TARGET = 110.0
    MIN_ZOOM = 1.0e-8
    MAX_ZOOM = 8.0
    DETAILS_LOD = 0.72
    CELLS_LOD = 0.18

    def __init__(
        self,
        width: int = 1,
        height: int = 1,
        orientation: GridOrientation | str = GridOrientation.Y_DOWN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._graphics_scene = QGraphicsScene(self)
        self._graphics_scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setScene(self._graphics_scene)
        self.setObjectName("frameMatrixView")
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor("#0b1120"))

        self.matrix_width = 1
        self.matrix_height = 1
        self._orientation = GridOrientation.Y_DOWN
        self._cells: dict[tuple[int, int], FrameCellData] = {}
        self._status_colors = dict(DEFAULT_STATUS_COLORS)
        self._selection = FrameSelection()
        self._selection_anchor: tuple[int, int] | None = None
        self._drag_anchor: tuple[int, int] | None = None
        self._drag_base_rectangles: tuple[FrameRect, ...] = ()
        self._panning = False
        self._pan_position = QPoint()
        self._visible_tiles: dict[tuple[int, int], _ViewportTileItem] = {}
        self._tile_index: dict[tuple[int, int], tuple[FrameCellData, ...]] = {}
        self._indexed_tile_cells = 0
        self._last_lod: MatrixLod | None = None
        self._update_scheduled = False

        self.horizontalScrollBar().valueChanged.connect(self._schedule_visible_update)
        self.verticalScrollBar().valueChanged.connect(self._schedule_visible_update)
        self.set_matrix_size(width, height, orientation)

    @staticmethod
    def _coerce_orientation(value: GridOrientation | str) -> GridOrientation:
        try:
            return GridOrientation(str(value))
        except ValueError as exc:
            raise ValueError(f"unsupported grid orientation: {value!r}") from exc

    def set_matrix_size(
        self,
        width: int,
        height: int,
        orientation: GridOrientation | str | None = None,
    ) -> None:
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("matrix dimensions must be positive")
        self.matrix_width = int(width)
        self.matrix_height = int(height)
        if orientation is not None:
            self._orientation = self._coerce_orientation(orientation)
        self._cells = {
            coordinate: cell
            for coordinate, cell in self._cells.items()
            if cell.x <= self.matrix_width and cell.y <= self.matrix_height
        }
        self._selection = self._clamp_selection(self._selection)
        self._graphics_scene.setSceneRect(
            0.0,
            0.0,
            self.matrix_width * self.CELL_PITCH,
            self.matrix_height * self.CELL_PITCH,
        )
        self._clear_tiles()
        self._indexed_tile_cells = 0
        self._update_visible_tiles()
        self.viewport().update()

    def matrix_size(self) -> tuple[int, int]:
        return self.matrix_width, self.matrix_height

    def orientation(self) -> GridOrientation:
        return self._orientation

    def set_orientation(self, orientation: GridOrientation | str) -> None:
        normalized = self._coerce_orientation(orientation)
        if normalized is self._orientation:
            return
        self._orientation = normalized
        self._indexed_tile_cells = 0
        self._invalidate_tiles()

    def set_cells(self, cells: Iterable[FrameCellData]) -> None:
        prepared: dict[tuple[int, int], FrameCellData] = {}
        for cell in cells:
            self._ensure_inside(cell.x, cell.y)
            prepared[(cell.x, cell.y)] = cell
        self._cells = prepared
        self._indexed_tile_cells = 0
        self._invalidate_tiles()

    def update_cells(self, cells: Iterable[FrameCellData]) -> None:
        for cell in cells:
            self._ensure_inside(cell.x, cell.y)
            self._cells[(cell.x, cell.y)] = cell
        self._indexed_tile_cells = 0
        self._invalidate_tiles()

    def remove_cells(self, coordinates: Iterable[tuple[int, int]]) -> None:
        for x, y in coordinates:
            self._cells.pop((int(x), int(y)), None)
        self._indexed_tile_cells = 0
        self._invalidate_tiles()

    def clear_cells(self) -> None:
        self._cells.clear()
        self._indexed_tile_cells = 0
        self._invalidate_tiles()

    def cell_data(self, x: int, y: int) -> FrameCellData:
        self._ensure_inside(x, y)
        return self._cells.get((int(x), int(y)), FrameCellData(int(x), int(y)))

    def materialized_cell_count(self) -> int:
        return len(self._cells)

    def set_status_colors(self, colors: Mapping[str, str | QColor]) -> None:
        for status, color in colors.items():
            candidate = QColor(color)
            if not candidate.isValid():
                raise ValueError(f"invalid color for status {status!r}: {color!r}")
            self._status_colors[str(status)] = candidate.name(QColor.NameFormat.HexArgb)
        self._invalidate_tiles()

    def selection(self) -> FrameSelection:
        return self._selection

    def set_selection(self, selection: FrameSelection | FrameRect | None) -> None:
        if selection is None:
            normalized = FrameSelection()
        elif isinstance(selection, FrameRect):
            normalized = FrameSelection((selection,))
        elif isinstance(selection, FrameSelection):
            normalized = selection
        else:
            raise TypeError("selection must be FrameSelection, FrameRect or None")
        normalized = self._clamp_selection(normalized)
        if normalized == self._selection:
            return
        self._selection = normalized
        if self._drag_anchor is None:
            if normalized.rectangles:
                last = normalized.rectangles[-1]
                self._selection_anchor = (last.x1, last.y1)
            else:
                self._selection_anchor = None
        self._invalidate_tiles()
        self.selectionChanged.emit(normalized)

    def clear_selection(self) -> None:
        self.set_selection(None)

    def selected_coordinates(self, *, maximum: int | None = None) -> tuple[tuple[int, int], ...]:
        return tuple(self._selection.coordinates(maximum=maximum))

    def scene_rect_for_frame(self, x: int, y: int) -> QRectF:
        self._ensure_inside(x, y)
        column = int(x) - 1
        row = self._row_for_y(int(y))
        return QRectF(
            column * self.CELL_PITCH,
            row * self.CELL_PITCH,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )

    def frame_at_scene_pos(self, position: QPointF) -> tuple[int, int] | None:
        if position.x() < 0.0 or position.y() < 0.0:
            return None
        column = int(position.x() // self.CELL_PITCH)
        row = int(position.y() // self.CELL_PITCH)
        if column >= self.matrix_width or row >= self.matrix_height:
            return None
        if position.x() - column * self.CELL_PITCH > self.CELL_SIZE:
            return None
        if position.y() - row * self.CELL_PITCH > self.CELL_SIZE:
            return None
        return column + 1, self._y_for_row(row)

    def frame_at_viewport_pos(self, position: QPoint | QPointF) -> tuple[int, int] | None:
        point = position.toPoint() if isinstance(position, QPointF) else position
        return self.frame_at_scene_pos(self.mapToScene(point))

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
        rect = tile.boundingRect().adjusted(0.8, 0.8, -1.2, -1.2)
        painter.fillRect(rect, self._status_color(highest_status))
        border = QColor("#475569")
        tile_frames = self._tile_frame_rect(tile)
        if any(rectangle.intersects(tile_frames) for rectangle in self._selection.rectangles):
            border = QColor("#f8fafc")
        pen = QPen(border, max(1.0, 1.5 / max(self.zoom_factor(), self.MIN_ZOOM)))
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(rect)

    def _paint_tile_cells(self, painter: QPainter, tile: _ViewportTileItem, lod: MatrixLod) -> None:
        start_column = tile.tile_x * tile.tile_cells
        start_row = tile.tile_y * tile.tile_cells
        end_column = min(self.matrix_width, start_column + tile.tile_cells)
        end_row = min(self.matrix_height, start_row + tile.tile_cells)
        base_pen = QPen(QColor("#475569"), 1.0)
        base_pen.setCosmetic(True)
        selection_pen = QPen(QColor("#f8fafc"), 2.2)
        selection_pen.setCosmetic(True)
        for row in range(start_row, end_row):
            y = self._y_for_row(row)
            for column in range(start_column, end_column):
                x = column + 1
                cell = self._cells.get((x, y), FrameCellData(x, y))
                rect = QRectF(
                    column * self.CELL_PITCH,
                    row * self.CELL_PITCH,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                painter.fillRect(rect, self._status_color(cell.status))
                if lod is MatrixLod.DETAILS and cell.thumbnail is not None and not cell.thumbnail.isNull():
                    painter.save()
                    painter.setOpacity(0.62)
                    painter.drawPixmap(rect.toRect(), cell.thumbnail)
                    painter.restore()

                performer_pen: QPen | None = None
                if cell.performer_color:
                    performer_color = QColor(cell.performer_color)
                    if performer_color.isValid():
                        performer_pen = QPen(performer_color, 3.0)
                        performer_pen.setCosmetic(True)
                painter.setPen(
                    selection_pen
                    if self._selection.contains(x, y)
                    else performer_pen or base_pen
                )
                painter.drawRect(rect)

                if lod is MatrixLod.DETAILS:
                    painter.setPen(QColor("#f8fafc"))
                    coordinate_text = cell.label or f"{x}, {y}"
                    painter.drawText(
                        rect.adjusted(4.0, 3.0, -4.0, -3.0),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                        coordinate_text,
                    )
                    if cell.performer_initials:
                        painter.drawText(
                            rect.adjusted(4.0, 3.0, -4.0, -3.0),
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                            cell.performer_initials,
                        )

    def _after_transform(self) -> None:
        self._indexed_tile_cells = 0
        self._update_visible_tiles()
        self._invalidate_tiles()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if not delta:
            super().wheelEvent(event)
            return
        current = self.zoom_factor()
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, current * math.pow(1.0015, delta)))
        if not math.isclose(current, target):
            factor = target / current
            self.scale(factor, factor)
            self._after_transform()
        event.accept()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._schedule_visible_update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() is Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_position = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() is Qt.MouseButton.LeftButton:
            coordinate = self.frame_at_viewport_pos(event.position())
            if coordinate is not None:
                modifiers = event.modifiers()
                if modifiers & Qt.KeyboardModifier.ShiftModifier and self._selection_anchor:
                    self._drag_anchor = self._selection_anchor
                    self._drag_base_rectangles = ()
                else:
                    self._drag_anchor = coordinate
                    self._selection_anchor = coordinate
                    self._drag_base_rectangles = (
                        self._selection.rectangles
                        if modifiers & Qt.KeyboardModifier.ControlModifier
                        else ()
                    )
                self.set_selection(
                    FrameSelection(
                        self._drag_base_rectangles
                        + (FrameRect.from_points(self._drag_anchor, coordinate),)
                    )
                )
                event.accept()
                return
            if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.clear_selection()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._panning:
            current = event.position().toPoint()
            delta = current - self._pan_position
            self._pan_position = current
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._drag_anchor is not None and event.buttons() & Qt.MouseButton.LeftButton:
            coordinate = self.frame_at_viewport_pos(event.position())
            if coordinate is not None:
                self.set_selection(
                    FrameSelection(
                        self._drag_base_rectangles
                        + (FrameRect.from_points(self._drag_anchor, coordinate),)
                    )
                )
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() is Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            self._schedule_visible_update()
            event.accept()
            return
        if event.button() is Qt.MouseButton.LeftButton and self._drag_anchor is not None:
            self._drag_anchor = None
            self._drag_base_rectangles = ()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() is Qt.MouseButton.LeftButton:
            coordinate = self.frame_at_viewport_pos(event.position())
            if coordinate is not None:
                self.frameActivated.emit(*coordinate)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # type: ignore[override]
        coordinate = self.frame_at_viewport_pos(event.pos())
        if coordinate is None:
            event.ignore()
            return
        x, y = coordinate
        if not self._selection.contains(x, y):
            self.set_selection(FrameSelection.single(x, y))
        context = FrameContext(x, y, self.cell_data(x, y), self._selection)
        self.contextMenuRequested.emit(context, event.globalPos())
        event.accept()
