"""qt: construction."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable, Mapping
from PyQt6.QtCore import QPoint, QPointF, QRectF
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)
from kraken_core.frame_matrix.models import (
    MatrixAggregate,
)

from kraken_core.frame_matrix.qt_parts.module_state import (
    DEFAULT_STATUS_COLORS,
    GridOrientation,
)

from kraken_core.frame_matrix.qt_parts.frame_rect import FrameRect

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import (
    FrameCellData,
    FrameSelection,
)


class ConstructionMixin:
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

        self.matrix_width = 1
        self.matrix_height = 1
        self._orientation = GridOrientation.Y_DOWN
        self._cells: dict[tuple[int, int], FrameCellData] = {}
        self._status_colors = dict(DEFAULT_STATUS_COLORS)
        self._border_mode = "status"
        self._fill_mode = "thumbnail"
        self._selection = FrameSelection()
        self._selection_anchor: tuple[int, int] | None = None
        self._drag_anchor: tuple[int, int] | None = None
        self._drag_base_rectangles: tuple[FrameRect, ...] = ()
        self._panning = False
        self._pan_position = QPoint()
        self._visible_tiles: dict[tuple[int, int], _ViewportTileItem] = {}
        self._tile_index: dict[tuple[int, int], tuple[FrameCellData, ...]] = {}
        self._aggregates: tuple[MatrixAggregate, ...] = ()
        self._indexed_tile_cells = 0
        self._last_lod: MatrixLod | None = None
        self._update_scheduled = False

        self.horizontalScrollBar().valueChanged.connect(self._schedule_visible_update)
        self.verticalScrollBar().valueChanged.connect(self._schedule_visible_update)
        self.apply_kraken_scene_theme()
        self.set_matrix_size(width, height, orientation)

    def apply_kraken_scene_theme(self) -> None:
        from kraken_core.theme import scene_color

        self.setBackgroundBrush(QColor(scene_color("matrix")))
        self._status_colors["empty"] = scene_color("empty")
        if getattr(self, "_visible_tiles", None):
            self._invalidate_tiles()
        elif self.viewport() is not None:
            self.viewport().update()

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
        self._emit_cells_changed()

    def set_aggregates(self, aggregates: Iterable[MatrixAggregate]) -> None:
        self._aggregates = tuple(aggregates)
        self._invalidate_tiles()

    def update_cells(self, cells: Iterable[FrameCellData]) -> None:
        for cell in cells:
            self._ensure_inside(cell.x, cell.y)
            self._cells[(cell.x, cell.y)] = cell
        self._indexed_tile_cells = 0
        self._invalidate_tiles()
        self._emit_cells_changed()

    def remove_cells(self, coordinates: Iterable[tuple[int, int]]) -> None:
        for x, y in coordinates:
            self._cells.pop((int(x), int(y)), None)
        self._indexed_tile_cells = 0
        self._invalidate_tiles()

    def clear_cells(self) -> None:
        self._cells.clear()
        self._aggregates = ()
        self._indexed_tile_cells = 0
        self._invalidate_tiles()
        self._emit_cells_changed()

    def cell_data(self, x: int, y: int) -> FrameCellData:
        self._ensure_inside(x, y)
        return self._cells.get((int(x), int(y)), FrameCellData(int(x), int(y)))

    def materialized_cell_count(self) -> int:
        return len(self._cells)

    def _emit_cells_changed(self) -> None:
        signal = getattr(self, "cellsChanged", None)
        if signal is not None:
            signal.emit()

    def legend_rows(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for cell in self._cells.values():
            payload = cell.payload if isinstance(cell.payload, Mapping) else {}
            row: dict[str, object] = {
                "performer_initials": str(payload.get("performer_initials") or cell.performer_initials or ""),
                "performer_color": str(payload.get("performer_color") or cell.performer_color or ""),
            }
            if payload.get("performer_name"):
                row["performer_name"] = str(payload.get("performer_name"))
            if "modified_at" in payload:
                row["modified_at"] = payload.get("modified_at")
            if "quality" in payload:
                row["quality"] = payload.get("quality")
            if "review_status" in payload and payload.get("review_status") not in ("", None):
                row["review_status"] = payload.get("review_status")
            elif cell.status:
                row["status"] = cell.status
            rows.append(row)
        return tuple(rows)

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

if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt_parts.matrix_lod import MatrixLod
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import _ViewportTileItem
