"""navigation_mixin: handle frame matrix arrow."""
from __future__ import annotations
from typing import Any
from PyQt6.QtGui import QKeyEvent
from ...adapters.qt.object_validity import qt_object_is_valid
from contour.widget_parts._imports import (
    QEvent,
)


class HandleFrameMatrixArrowMixin:
    def _handle_frame_matrix_arrow_key_event(self: Any, event: object) -> bool:
        from ...graphics.editor_hotkeys import event_matches_shortcut

        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return False
        if getattr(self, "_closing", False) or not qt_object_is_valid(self.thumbnail_grid):
            return False
        if not isinstance(event, QKeyEvent) or event.type() != QEvent.Type.KeyPress:
            return False
        columns = max(1, self._thumbnail_columns())
        delta = next(
            (
                value
                for action, default, value in (
                    ("frame_previous", "Ctrl+Left", -1),
                    ("frame_next", "Ctrl+Right", 1),
                    ("frame_previous_row", "Ctrl+Up", -columns),
                    ("frame_next_row", "Ctrl+Down", columns),
                )
                if event_matches_shortcut(event, action, default)
            ),
            None,
        )
        if delta is None:
            return False
        count = int(self.thumbnail_grid.count())
        if count <= 0:
            event.accept()
            return True
        current_row: int | None = int(self.thumbnail_grid.currentRow())
        if current_row is not None and current_row < 0:
            current_path = getattr(self._workspace, "current_image_path", None)
            current_row = self._thumbnail_row_for_path(str(current_path)) if current_path else None
        if current_row is None or current_row < 0:
            current_row = 0
        target_row = max(0, min(count - 1, int(current_row) + delta))
        step = 1 if delta > 0 else -1
        while 0 <= target_row < count:
            item = self.thumbnail_grid.item(target_row)
            if item is not None and not item.isHidden():
                break
            target_row += step
        if target_row < 0 or target_row >= count:
            event.accept()
            return True
        if target_row != current_row:
            self.thumbnail_grid.setCurrentRow(target_row)
            self._scroll_thumbnail_grid_to_row(target_row)
            self._on_frame_navigation_requested(target_row)
        event.accept()
        return True
