"""qt: cell visual color."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent

from kraken_core.frame_matrix.qt_parts.frame_rect import FrameRect

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData, FrameSelection


@dataclass(frozen=True, slots=True)
class PreparedCellVisual:
    """Border and fill already resolved for one materialized cell."""

    x: int
    y: int
    border: QColor
    fill: QColor


class CellVisualColorMixin:
    def set_visual_modes(self, border: str, fill: str) -> None:
        border_modes = {"none", "time", "performer", "quality", "status"}
        fill_modes = {"time", "performer", "quality", "status", "thumbnail"}
        if border not in border_modes or fill not in fill_modes:
            raise ValueError("Unsupported matrix visual mode")
        self._border_mode, self._fill_mode = border, fill
        self._store_cell_visuals(None)
        self._invalidate_tiles()

    def visual_modes(self) -> tuple[str, str]:
        return self._border_mode, self._fill_mode

    def border_colors(self) -> Mapping[tuple[int, int], QColor]:
        return self._border_colors

    def fill_colors(self) -> Mapping[tuple[int, int], QColor]:
        return self._fill_colors

    def cell_visuals(self) -> tuple[PreparedCellVisual, ...]:
        """Return the stored border/fill array. This does not resolve colors."""
        return self._prepared_visuals

    def _store_cell_visuals(self, cells: Iterable[FrameCellData] | None) -> None:
        """Resolve border and fill for changed cells and publish them as one array."""
        self._refresh_time_span()
        if cells is None or self._border_mode == "time" or self._fill_mode == "time":
            self._border_colors = {}
            self._fill_colors = {}
            targets = tuple(self._cells.values())
        else:
            targets = tuple(cells)
        for cell in targets:
            key = (cell.x, cell.y)
            self._border_colors[key] = self._cell_visual_color(cell, self._border_mode)
            self._fill_colors[key] = self._cell_visual_color(cell, self._fill_mode)
        sample = FrameCellData(1, 1)
        self._empty_border = self._cell_visual_color(sample, self._border_mode)
        self._empty_fill = self._cell_visual_color(sample, self._fill_mode)
        prepared = {
            key: PreparedCellVisual(key[0], key[1], self._border_colors[key], self._fill_colors[key])
            for key in self._border_colors
        }
        self._prepared_by_cell = prepared
        self._prepared_visuals = tuple(prepared.values())

    def _refresh_time_span(self) -> None:
        """Widen the layer time scale. The visible window and the selection do not shrink it."""

        self._time_span_builds += 1
        stamps = [
            parsed
            for value in self._cells.values()
            if (
                parsed := self._timestamp(
                    (value.payload if isinstance(value.payload, Mapping) else {}).get("modified_at")
                )
            )
            is not None
        ]
        if not stamps:
            return
        oldest, newest = min(stamps), max(stamps)
        current = self._time_span
        if current is None:
            self._time_span = (oldest, newest)
            return
        self._time_span = (min(current[0], oldest), max(current[1], newest))

    def _cell_visual_color(self, cell: FrameCellData, mode: str) -> QColor:
        payload = cell.payload if isinstance(cell.payload, Mapping) else {}
        neutral = QColor("#475569")
        if mode == "none":
            return QColor()
        if mode == "thumbnail":
            return self._status_color(cell.status)
        if mode == "performer":
            color = QColor(str(payload.get("performer_color") or cell.performer_color or ""))
            return color if color.isValid() else neutral
        if mode == "quality":
            try:
                value = max(0.0, min(1.0, float(payload["quality"])))
            except (KeyError, TypeError, ValueError):
                return neutral
            return QColor.fromRgbF(1.0 - value, value, 0.10)
        if mode == "status":
            return {
                "not_checked": QColor("#64748b"),
                "in_review": QColor("#f59e0b"),
                "checked": QColor("#22c55e"),
            }.get(str(payload.get("review_status") or ""), self._status_color(cell.status))
        if mode == "time":
            stamp = self._timestamp(payload.get("modified_at"))
            span = self._time_span
            if stamp is None or span is None:
                return neutral
            oldest, newest = span
            ratio = 1.0 if newest <= oldest else (stamp - oldest) / (newest - oldest)
            return QColor.fromRgbF(1.0 - ratio, ratio, 0.08)
        return neutral

    @staticmethod
    def _timestamp(value: object) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    def _after_transform(self) -> None:
        self._indexed_tile_cells = 0
        self._update_visible_tiles()
        self._invalidate_tiles()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        blocked_modifiers = (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if not event.modifiers() & blocked_modifiers and (
            event.key() == Qt.Key.Key_F
            or event.text().casefold() in {"f", "а"}
        ):
            self.zoom_to_fit()
            event.accept()
            return
        super().keyPressEvent(event)

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
                self._begin_layer_selection_profile()
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
            self._finish_layer_selection_profile_after_paint()
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

    def _begin_layer_selection_profile(self) -> None:
        from kraken_core.frame_matrix import profiling

        current = getattr(self, "_layer_selection_profile", None)
        if current is not None:
            current.finish()
        self._layer_selection_profile = profiling.begin_region("layer_selection")

    def _finish_layer_selection_profile_after_paint(self) -> None:
        session = getattr(self, "_layer_selection_profile", None)
        if session is None:
            return
        self._layer_selection_profile = None
        from PyQt6.QtCore import QEvent, QObject, QTimer

        viewport = self.viewport()

        class _Ready(QObject):
            def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
                if watched is viewport and event.type() == QEvent.Type.Paint:
                    viewport.removeEventFilter(self)
                    QTimer.singleShot(0, lambda: session.finish(ready=True))
                return False

        ready = _Ready(self)
        self._layer_selection_ready = ready
        viewport.installEventFilter(ready)
        viewport.update()

if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameCellData
