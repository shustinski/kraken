"""pipeline_mixin: remove selected color entry."""
from __future__ import annotations
from contour.widget_parts._imports import (
    QListWidget,
    QListWidgetItem,
    Qt,
)


class RemoveSelectedColorEntryMixin:
    def _on_color_entry_changed(self, row: int, color_list: QListWidget, item: QListWidgetItem) -> None:
        entries = self._color_selection_entries(row)
        index = color_list.row(item)
        if index < 0 or index >= len(entries):
            return
        entries[index]["enabled"] = item.checkState() == Qt.CheckState.Checked
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._auto_apply_pipeline()

    def _remove_selected_color_entry(self, row: int, color_list: QListWidget) -> None:
        index = color_list.currentRow()
        if index < 0:
            return
        entries = self._color_selection_entries(row)
        if index >= len(entries):
            return
        entries.pop(index)
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _clear_color_entries(self, row: int) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters["selected_colors"] = []
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _set_color_pick_active(self, row: int | None) -> None:
        self._color_pick_pipeline_row = row
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_click_mode(row is not None)

    def _set_via_template_pick_active(self, enabled: bool) -> None:
        if enabled:
            self._set_color_pick_active(None)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_region_selection_mode(enabled)
