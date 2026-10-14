"""editor_view: wheel event."""
from __future__ import annotations
from PyQt6.QtCore import (
    QPointF,
    Qt,
)
from PyQt6.QtGui import (
    QCloseEvent,
    QWheelEvent,
)
from contour.graphics.viewport_navigation import (
    zoom_factor_for_wheel_delta,
)


class WheelEventMixin:
    def _update_paste_preview(self, scene_pos: QPointF | None) -> None:
        if not self._paste_mode or scene_pos is None:
            return
        if len(self._paste_preview_items) != len(self._clipboard_polygons):
            self._rebuild_paste_preview()
        dx = scene_pos.x() - self._clipboard_anchor.x()
        dy = scene_pos.y() - self._clipboard_anchor.y()
        for item in self._paste_preview_items:
            item.setPos(dx, dy)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        viewport_point = self._wheel_event_viewport_pixel(event)
        delta = event.angleDelta()
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if delta.y() == 0:
                event.accept()
                return
            self._queue_wheel_zoom(viewport_point, zoom_factor_for_wheel_delta(delta.y()))
            event.accept()
            return
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            delta_value = delta.x() if delta.x() else delta.y()
            scrollbar = self.horizontalScrollBar()
            if scrollbar is not None:
                scrollbar.setValue(scrollbar.value() - delta_value)
            event.accept()
            return
        super().wheelEvent(event)
        self._update_tool_cursors()
        event.accept()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._wheel_zoom_timer.stop()
        self._pending_wheel_zoom_factor = 1.0
        self._pending_wheel_zoom_viewport_pixel = None
        self._stop_zoom_animation()
        self._gradient_arrows_timer.stop()
        if event is not None:
            super().closeEvent(event)
