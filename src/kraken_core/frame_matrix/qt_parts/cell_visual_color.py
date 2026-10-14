"""qt: cell visual color."""
from __future__ import annotations
import math
from collections.abc import Mapping
from datetime import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent, QWheelEvent

from kraken_core.frame_matrix.qt_parts.frame_rect import FrameRect

from kraken_core.frame_matrix.qt_parts.viewport_tile_item import FrameSelection


class CellVisualColorMixin:
    def set_visual_modes(self, border: str, fill: str) -> None:
        border_modes = {"time", "performer", "quality", "status"}
        fill_modes = border_modes | {"thumbnail"}
        if border not in border_modes or fill not in fill_modes:
            raise ValueError("Unsupported matrix visual mode")
        self._border_mode, self._fill_mode = border, fill
        self.viewport().update()

    def visual_modes(self) -> tuple[str, str]:
        return self._border_mode, self._fill_mode

    def _cell_visual_color(self, cell: FrameCellData, mode: str) -> QColor:
        payload = cell.payload if isinstance(cell.payload, Mapping) else {}
        neutral = QColor("#475569")
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
            values = [
                parsed
                for value in self._cells.values()
                if (parsed := self._timestamp(
                    (value.payload if isinstance(value.payload, Mapping) else {}).get("modified_at")
                )) is not None
            ]
            if stamp is None or not values:
                return neutral
            oldest, newest = min(values), max(values)
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

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if not delta:
            super().wheelEvent(event)
            return
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            current = self.zoom_factor()
            target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, current * math.pow(1.0015, delta)))
            if not math.isclose(current, target):
                factor = target / current
                self.scale(factor, factor)
                self._after_transform()
        else:
            scrollbar = (
                self.horizontalScrollBar()
                if modifiers & Qt.KeyboardModifier.ShiftModifier
                else self.verticalScrollBar()
            )
            pixel_delta = event.pixelDelta()
            amount = pixel_delta.y() if not pixel_delta.isNull() else delta / 8.0
            scrollbar.setValue(scrollbar.value() - round(amount))
        event.accept()

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
