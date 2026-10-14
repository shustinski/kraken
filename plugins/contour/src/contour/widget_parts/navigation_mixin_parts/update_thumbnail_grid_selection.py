"""navigation_mixin: update thumbnail grid selection."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QBrush,
    QColor,
    Qt,
)


class UpdateThumbnailGridSelectionMixin:
    def _update_thumbnail_grid_selection(self: Any, *, scroll_to_selection: bool | None = None) -> None:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        if scroll_to_selection is None:
            scroll_to_selection = getattr(self, "_suppress_thumbnail_grid_scroll_path", None) is None
        # Selection leads the asynchronous workspace load.  Highlight the
        # requested frame immediately, while generation checks still prevent
        # stale load results from changing the editor.
        current = getattr(self, "_desired_image_path", None) or self._workspace.current_image_path
        previous = getattr(self, "_thumbnail_selected_path", None)
        path_index = getattr(self, "_thumbnail_path_to_row", {})
        matched_index = -1
        previous_item = None
        current_item = None
        if previous:
            previous_row = path_index.get(self._normalized_path(previous))
            if previous_row is not None:
                previous_item = self.thumbnail_grid.item(previous_row)
        if current:
            matched_index = path_index.get(self._normalized_path(current), -1)
            if matched_index < 0:
                for index in range(self.thumbnail_grid.count()):
                    item = self.thumbnail_grid.item(index)
                    if item is None:
                        continue
                    item_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                    if item_path and self._normalized_path(item_path) == self._normalized_path(current):
                        matched_index = index
                        break
            if matched_index >= 0:
                current_item = self.thumbnail_grid.item(matched_index)
        self.thumbnail_grid.blockSignals(True)
        try:
            if previous_item is not None:
                previous_item.setBackground(QBrush())
                previous_item.setData(Qt.ItemDataRole.BackgroundRole, None)
            if current_item is not None and matched_index >= 0 and not current_item.isHidden():
                self.thumbnail_grid.setCurrentRow(matched_index)
                if hasattr(self.thumbnail_grid, "setCurrentFrameId"):
                    self.thumbnail_grid.setCurrentFrameId(matched_index)
                if hasattr(self, "polygon_editor"):
                    self.polygon_editor.set_current_frame_id(matched_index, center=False, emit_signal=False)
                current_item.setBackground(QBrush(QColor("#1D4ED8")))
                current_item.setData(Qt.ItemDataRole.BackgroundRole, QColor("#1D4ED8"))
                self._thumbnail_selected_path = str(current)
            else:
                self.thumbnail_grid.clearSelection()
                self.thumbnail_grid.setCurrentRow(-1)
                self._thumbnail_selected_path = None
        finally:
            self.thumbnail_grid.blockSignals(False)
        if matched_index >= 0 and scroll_to_selection:
            self._scroll_thumbnail_grid_to_row(matched_index)
        suppressed_path = getattr(self, "_suppress_thumbnail_grid_scroll_path", None)
        if suppressed_path and current and str(Path(current)) == str(Path(suppressed_path)):
            self._suppress_thumbnail_grid_scroll_path = None
        current_path = self._normalized_path(current) if current else None
        if (
            current_path
            and current_path != getattr(self, "_thumbnail_radial_center_path", None)
            and not self._thumbnail_loading_blocked()
        ):
            self._schedule_visible_thumbnail_loads()
