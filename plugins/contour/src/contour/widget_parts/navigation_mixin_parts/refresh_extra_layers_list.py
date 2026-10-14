"""navigation_mixin: refresh extra layers list."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QColor,
    QColorDialog,
    QListWidgetItem,
    QPushButton,
    Qt,
    normalize_via_display_mode,
)

from contour.widget_parts.navigation_mixin_parts.module_state import _LOGGER


class RefreshExtraLayersListMixin:
    def _apply_cif_directory_edit(self: Any) -> None:
        path = self.cif_dir_edit.text().strip()
        if path:
            self.set_cif_directory(path)
        else:
            self._vector_indexer.invalidate_pending_results()
            self._workspace.clear_cif_index()
            self._save_persisted_paths()
            self._rebuild_vector_list()
            self._refresh_image_list_item_states()
            self._rebuild_asset_filter_lists()
            self._apply_asset_view_filter()
            if self._workspace.current_image_path:
                try:
                    self.load_image(self._workspace.current_image_path)
                except Exception as exc:
                    _LOGGER.exception(
                        "Failed to reload image %s after directory change",
                        self._workspace.current_image_path,
                    )
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _apply_output_directory_edit(self: Any) -> None:
        self.set_output_directory(self.output_dir_edit.text().strip())

    def _apply_dataset_directory_edit(self: Any) -> None:
        self.set_dataset_directory(self.dataset_dir_edit.text().strip())

    def _choose_external_color(self: Any) -> None:
        self._choose_color("external_color", self.external_color_button)

    def _choose_hole_color(self: Any) -> None:
        self._choose_color("hole_color", self.hole_color_button)

    def _choose_selected_color(self: Any) -> None:
        self._choose_color("selected_color", self.selected_color_button)

    def _choose_via_selection_color(self: Any) -> None:
        self._choose_color("via_selection_color", self.via_selection_color_button)

    def _choose_conductor_hover_highlight_color(self: Any) -> None:
        self._choose_color("conductor_hover_highlight_color", self.conductor_hover_highlight_color_button)

    def _choose_vertex_color(self: Any) -> None:
        self._choose_color("vertex_color", self.vertex_color_button)

    def _choose_color(self: Any, attribute_name: str, button: QPushButton) -> None:
        initial = QColor(getattr(self._display_settings, attribute_name))
        color = QColorDialog.getColor(initial, self, self._tr("select_color_dialog_title"))
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexRgb)
        setattr(self._display_settings, attribute_name, value)
        self._update_color_button(button, value)
        self._apply_display_settings()

    def _apply_display_settings(self: Any) -> None:
        if hasattr(self, "line_width_spin"):
            self._display_settings.line_width = float(self.line_width_spin.value())
            self._display_settings.vertex_size = float(self.vertex_size_spin.value())
            self._display_settings.fill_opacity = float(self.fill_opacity_spin.value())
            self._display_settings.show_vertices = bool(self.show_vertices_checkbox.isChecked())
            self._display_settings.show_labels = bool(self.show_labels_checkbox.isChecked())
            if hasattr(self, "via_display_mode_combo"):
                self._display_settings.via_display_mode = normalize_via_display_mode(
                    self.via_display_mode_combo.currentData()
                )
            if hasattr(self, "polygon_editor"):
                self.polygon_editor.set_display_settings(self._display_settings)
                if hasattr(self, "random_object_colors_checkbox"):
                    self.polygon_editor.set_random_object_colors_enabled(self.random_object_colors_checkbox.isChecked())
        self._apply_vector_geometry_editor_config()
        self._save_persisted_display_settings()

    def _on_neighbor_display_settings_changed(self: Any, *_args) -> None:
        if not self._ensure_neighbor_pyramid_or_prompt():
            self._sync_neighbor_frames()
            self._configure_thumbnail_grid_geometry()
            self._save_persisted_display_settings()
            self._schedule_frame_prefetch()
            return
        self._sync_neighbor_frames()
        self._configure_thumbnail_grid_geometry()
        if hasattr(self, "_configure_pyramid_frame_store"):
            self._configure_pyramid_frame_store(
                [str(Path(path)) for path in getattr(self._workspace, "image_paths", [])]
            )
        self._save_persisted_display_settings()
        self._schedule_frame_prefetch()

    def _refresh_extra_layers_list(self: Any) -> None:
        if not hasattr(self, "extra_layers_list"):
            return
        current_item = self.extra_layers_list.currentItem()
        current_id = current_item.data(Qt.ItemDataRole.UserRole) if current_item is not None else None
        self.extra_layers_list.blockSignals(True)
        self.extra_layers_list.clear()
        for index, layer in enumerate(self._extra_layers):
            layer_id = int(layer.get("id", index + 1))
            folder_path = str(layer.get("folder_path", ""))
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, layer_id)
            item.setToolTip(folder_path)
            row_widget = self._build_extra_layer_row_widget(layer)
            item.setSizeHint(row_widget.sizeHint())
            self.extra_layers_list.addItem(item)
            self.extra_layers_list.setItemWidget(item, row_widget)
        self.extra_layers_list.blockSignals(False)
        if current_id is not None:
            for row in range(self.extra_layers_list.count()):
                candidate = self.extra_layers_list.item(row)
                if candidate is not None and candidate.data(Qt.ItemDataRole.UserRole) == current_id:
                    self.extra_layers_list.setCurrentRow(row)
                    break
