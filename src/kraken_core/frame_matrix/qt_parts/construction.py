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
    PreparedSelectionPaint,
    _SelectionOverlayItem,
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
        self._cell_w = float(self.CELL_SIZE)
        self._cell_h = float(self.CELL_SIZE)
        self._pitch_x = float(self.CELL_PITCH)
        self._pitch_y = float(self.CELL_PITCH)
        self._graphics_scene = QGraphicsScene(self)
        self._graphics_scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setScene(self._graphics_scene)
        self.setObjectName("frameMatrixView")
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self._configure_matrix_viewport()

        self.matrix_width = 1
        self.matrix_height = 1
        self._orientation = GridOrientation.Y_DOWN
        self._cells: dict[tuple[int, int], FrameCellData] = {}
        self._status_colors = dict(DEFAULT_STATUS_COLORS)
        self._border_mode = "status"
        self._fill_mode = "thumbnail"
        self._border_colors: dict[tuple[int, int], QColor] = {}
        self._fill_colors: dict[tuple[int, int], QColor] = {}
        self._prepared_by_cell: dict[tuple[int, int], object] = {}
        self._prepared_visuals: tuple = ()
        self._empty_border = QColor("#475569")
        self._empty_fill = QColor("#475569")
        self._time_span: tuple[float, float] | None = None
        self._time_span_builds = 0
        self._selection = FrameSelection()
        self._selection_fill_rects: tuple[QRectF, ...] = ()
        self._selection_border_rects: tuple[QRectF, ...] = ()
        self._selection_paints: tuple[PreparedSelectionPaint, ...] = ()
        self._selected_coordinates: frozenset[tuple[int, int]] = frozenset()
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
        self._selection_overlay = _SelectionOverlayItem(self)
        self._graphics_scene.addItem(self._selection_overlay)
        self.set_matrix_size(width, height, orientation)

    def apply_kraken_scene_theme(self) -> None:
        from kraken_core.theme import scene_color

        self.setBackgroundBrush(QColor(scene_color("matrix")))
        if hasattr(self, "resetCachedContent"):
            self.resetCachedContent()
        self._status_colors["empty"] = scene_color("empty")
        if hasattr(self, "_cells"):
            self._store_cell_visuals(None)
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
        overlay = getattr(self, "_selection_overlay", None)
        if overlay is not None:
            overlay.prepareGeometryChange()
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
        self._store_selection_paints()
        self._store_cell_visuals(None)
        self._apply_scene_extent()
        self._clear_tiles()
        self._indexed_tile_cells = 0
        self._update_visible_tiles()
        self.viewport().update()

    def cell_size(self) -> tuple[float, float]:
        return self._cell_w, self._cell_h

    def cell_pitch(self) -> tuple[float, float]:
        return self._pitch_x, self._pitch_y

    def set_cell_geometry(
        self,
        width: float,
        height: float,
        overlap_x: float = 0.0,
        overlap_y: float = 0.0,
    ) -> None:
        """Set the cell size and how far neighbouring cells overlap.

        An overlap is clamped below the cell size, so every cell keeps a
        positive step and later cells never collapse onto earlier ones.
        """

        cell_w = max(1.0, float(width))
        cell_h = max(1.0, float(height))
        overlap_x = min(max(0.0, float(overlap_x)), cell_w - 1.0)
        overlap_y = min(max(0.0, float(overlap_y)), cell_h - 1.0)
        pitch_x = max(1.0, cell_w + self.CELL_GAP - overlap_x)
        pitch_y = max(1.0, cell_h + self.CELL_GAP - overlap_y)
        if (cell_w, cell_h, pitch_x, pitch_y) == (self._cell_w, self._cell_h, self._pitch_x, self._pitch_y):
            return
        overlay = getattr(self, "_selection_overlay", None)
        if overlay is not None:
            overlay.prepareGeometryChange()
        self._cell_w, self._cell_h = cell_w, cell_h
        self._pitch_x, self._pitch_y = pitch_x, pitch_y
        self._apply_scene_extent()
        self._store_selection_paints()
        self._clear_tiles()
        self._indexed_tile_cells = 0
        self._update_visible_tiles()
        self.viewport().update()

    def _cell_rect(self, column: int, row: int) -> QRectF:
        """Scene rectangle of a zero-based column and row."""

        return QRectF(column * self._pitch_x, row * self._pitch_y, self._cell_w, self._cell_h)

    def _scene_extent(self, columns: int, rows: int) -> tuple[float, float]:
        """Scene size covered by ``columns`` x ``rows`` cells, overlap included."""

        width = 0.0 if columns <= 0 else (columns - 1) * self._pitch_x + max(self._cell_w, self._pitch_x)
        height = 0.0 if rows <= 0 else (rows - 1) * self._pitch_y + max(self._cell_h, self._pitch_y)
        return width, height

    def _apply_scene_extent(self) -> None:
        width, height = self._scene_extent(self.matrix_width, self.matrix_height)
        self._graphics_scene.setSceneRect(0.0, 0.0, width, height)

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
        self._store_selection_paints()
        self._invalidate_tiles()

    def set_cells(self, cells: Iterable[FrameCellData]) -> None:
        prepared: dict[tuple[int, int], FrameCellData] = {}
        for cell in cells:
            self._ensure_inside(cell.x, cell.y)
            prepared[(cell.x, cell.y)] = cell
        self._cells = prepared
        self._store_cell_visuals(None)
        self._indexed_tile_cells = 0
        self._invalidate_tiles()
        self._emit_cells_changed()

    def set_aggregates(self, aggregates: Iterable[MatrixAggregate]) -> None:
        self._aggregates = tuple(aggregates)
        self._invalidate_tiles()

    def update_cells(self, cells: Iterable[FrameCellData]) -> None:
        changed_tiles: dict[tuple[int, int], list[FrameCellData]] = {}
        changed: list[FrameCellData] = []
        for cell in cells:
            self._ensure_inside(cell.x, cell.y)
            self._cells[(cell.x, cell.y)] = cell
            changed.append(cell)
            if self._indexed_tile_cells:
                key = ((cell.x - 1) // self._indexed_tile_cells, self._row_for_y(cell.y) // self._indexed_tile_cells)
                changed_tiles.setdefault(key, []).append(cell)
        self._store_cell_visuals(changed)
        if self._indexed_tile_cells:
            for key, replacements in changed_tiles.items():
                indexed = {(cell.x, cell.y): cell for cell in self._tile_index.get(key, ())}
                indexed.update({(cell.x, cell.y): cell for cell in replacements})
                self._tile_index[key] = tuple(indexed.values())
                tile = self._visible_tiles.get(key)
                if tile is not None:
                    for cell in replacements:
                        column = cell.x - 1
                        row = self._row_for_y(cell.y)
                        tile.update(
                            self._cell_rect(column, row).adjusted(-4.0, -4.0, 4.0, 4.0)
                        )
        else:
            self._invalidate_tiles()
        self._emit_cells_changed()

    def remove_cells(self, coordinates: Iterable[tuple[int, int]]) -> None:
        for x, y in coordinates:
            key = (int(x), int(y))
            self._cells.pop(key, None)
            self._border_colors.pop(key, None)
            self._fill_colors.pop(key, None)
        self._store_cell_visuals(None if self._border_mode == "time" or self._fill_mode == "time" else ())
        self._indexed_tile_cells = 0
        self._invalidate_tiles()

    def clear_cells(self) -> None:
        self._cells.clear()
        self._aggregates = ()
        self._indexed_tile_cells = 0
        self._time_span = None
        self._store_cell_visuals(None)
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
        self._store_cell_visuals(None)
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
        self._store_selection_paints()
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

    def selection_fill_rects(self) -> tuple[QRectF, ...]:
        return self._selection_fill_rects

    def selection_border_rects(self) -> tuple[QRectF, ...]:
        return self._selection_border_rects

    def selection_paints(self) -> tuple[PreparedSelectionPaint, ...]:
        """Return the stored selection fill and frame array."""
        return self._selection_paints

    def _store_selection_paints(self) -> None:
        fills: list[QRectF] = []
        borders: list[QRectF] = []
        paints: list[PreparedSelectionPaint] = []
        selected: dict[tuple[int, int], None] = {}
        for rectangle in self._selection.rectangles:
            for x, y in rectangle.coordinates():
                if (x, y) in selected:
                    continue
                selected[(x, y)] = None
                column = x - 1
                row = self._row_for_y(y)
                rect = self._cell_rect(column, row)
                border = QRectF(rect)
                fills.append(rect)
                borders.append(border)
                paints.append(PreparedSelectionPaint(rect, border))
        self._selection_fill_rects = tuple(fills)
        self._selection_border_rects = tuple(borders)
        self._selection_paints = tuple(paints)
        self._selected_coordinates = frozenset(selected)

    def selected_coordinates(self, *, maximum: int | None = None) -> tuple[tuple[int, int], ...]:
        return tuple(self._selection.coordinates(maximum=maximum))

    def scene_rect_for_frame(self, x: int, y: int) -> QRectF:
        self._ensure_inside(x, y)
        column = int(x) - 1
        row = self._row_for_y(int(y))
        return self._cell_rect(column, row)

    def frame_at_scene_pos(self, position: QPointF) -> tuple[int, int] | None:
        if position.x() < 0.0 or position.y() < 0.0:
            return None
        column = min(int(position.x() // self._pitch_x), self.matrix_width - 1)
        row = min(int(position.y() // self._pitch_y), self.matrix_height - 1)
        if position.x() - column * self._pitch_x >= self._cell_w:
            return None
        if position.y() - row * self._pitch_y >= self._cell_h:
            return None
        return column + 1, self._y_for_row(row)

    def frame_at_viewport_pos(self, position: QPoint | QPointF) -> tuple[int, int] | None:
        point = position.toPoint() if isinstance(position, QPointF) else position
        return self.frame_at_scene_pos(self.mapToScene(point))


if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt_parts.matrix_lod import MatrixLod
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import _ViewportTileItem
