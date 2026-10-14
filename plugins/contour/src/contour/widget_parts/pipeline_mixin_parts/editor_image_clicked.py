"""pipeline_mixin: editor image clicked."""
from __future__ import annotations
from contour.widget_parts._imports import (
    np,
)


class EditorImageClickedMixin:
    def _add_color_selection(self, row: int, rgb: tuple[int, int, int]) -> None:
        entries = self._color_selection_entries(row)
        for entry in entries:
            if tuple(entry["rgb"]) == tuple(rgb):
                entry["enabled"] = True
                self._pipeline.steps[row].parameters["selected_colors"] = entries
                self._render_pipeline_parameters(row)
                self._auto_apply_pipeline()
                return
        entries.append({"rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])], "enabled": True})
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _on_editor_image_clicked(self, x_coord: float, y_coord: float) -> None:
        row = self._color_pick_pipeline_row
        if row is None or row < 0 or row >= len(self._pipeline.steps):
            return
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            return
        image = np.asarray(current_state.source_image)
        x_index = round(x_coord)
        y_index = round(y_coord)
        if y_index < 0 or x_index < 0 or y_index >= image.shape[0] or x_index >= image.shape[1]:
            return
        if image.ndim == 2:
            value = int(image[y_index, x_index])
            rgb = (value, value, value)
        else:
            pixel = image[y_index, x_index]
            if image.shape[2] >= 3:
                rgb = (int(pixel[2]), int(pixel[1]), int(pixel[0]))
            else:
                value = int(pixel[0])
                rgb = (value, value, value)
        self._add_color_selection(row, rgb)
        self._append_log(
            self._tr(
                "color_picked_log",
                "Добавлен цвет {color}" if self._ui_language == "ru" else "Added color {color}",
                color=f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
            )
        )
