"""matrix_view: mouse release event."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QToolTip,
)


class MouseReleaseEventMixin:
    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_active:
            self._pan_active = False
            self._pan_start = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover_scene_pos = None
        self._hover_prefetch_timer.stop()
        self._set_hover_item(None)
        QToolTip.hideText()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._clear_visible_record_key_cache()
        self._emit_overview_state()
        self._update_tile_lod()
