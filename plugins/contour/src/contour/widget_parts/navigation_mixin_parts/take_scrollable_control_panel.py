"""navigation_mixin: take scrollable control panel."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    THUMBNAIL_VISIBLE_LOAD_DEBOUNCE_MS,
    Path,
    QFrame,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    Qt,
    QWidget,
)


class TakeScrollableControlPanelMixin:
    def _apply_asset_view_filter(self: Any) -> None:
        match_only = bool(getattr(self, "_asset_filter_match_only", False))
        if hasattr(self, "image_list"):
            self._image_list_proxy.invalidateFilter()
            if not self._uses_large_frame_list():
                self._image_list_model.invalidate_all_rows()
        if self._frame_matrix_enabled() and hasattr(self, "thumbnail_grid"):
            for index in range(self.thumbnail_grid.count()):
                item = self.thumbnail_grid.item(index)
                if item is None:
                    continue
                path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                item.setHidden(bool(match_only and path and not self._image_path_has_matching_vector(path)))
            if not bool(getattr(self, "_thumbnail_rebuild_in_progress", False)):
                self._configure_thumbnail_grid_geometry()
                self._update_thumbnail_grid_selection()

    def _on_asset_image_item_clicked(self: Any, item: QListWidgetItem) -> None:
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        self._set_image_list_current_path(path, fallback_to_first=False)

    def _on_asset_vector_item_clicked(self: Any, item: QListWidgetItem) -> None:
        self._on_vector_item_navigate_request(item)

    def _take_thumbnail_matrix_panel(self: Any) -> QWidget:
        panel = self.thumbnail_matrix_panel
        if hasattr(self, "files_tab") and self.files_tab.layout() is not None:
            self.files_tab.layout().removeWidget(panel)
        return panel

    def _take_files_panel(self: Any) -> QWidget:
        panel = self.files_tab
        if hasattr(self, "right_tabs"):
            index = self.right_tabs.indexOf(panel)
            if index >= 0:
                self.right_tabs.removeTab(index)
            if self.right_tabs.count() == 0:
                self.right_tabs.hide()
                if hasattr(self, "main_splitter"):
                    self.right_tabs.setParent(None)
        return panel

    def _take_paths_panel(self: Any) -> QWidget:
        return self._take_scrollable_control_panel(
            self.paths_tab,
            object_name="pathsPanelScroll",
            attribute_name="paths_panel_scroll",
        )

    def _take_pipeline_panel(self: Any) -> QWidget:
        return self._take_scrollable_control_panel(
            self.pipeline_tab,
            object_name="pipelinePanelScroll",
            attribute_name="pipeline_panel_scroll",
        )

    def _take_display_panel(self: Any) -> QWidget:
        return self._take_scrollable_control_panel(
            self.display_tab,
            object_name="displayPanelScroll",
            attribute_name="display_panel_scroll",
        )

    def _take_recognition_panel(self: Any) -> QWidget:
        return self._take_scrollable_control_panel(
            self.extraction_tab,
            object_name="recognitionPanelScroll",
            attribute_name="recognition_panel_scroll",
        )

    def _take_scrollable_control_panel(
        self: Any,
        panel: QWidget,
        *,
        object_name: str,
        attribute_name: str,
    ) -> QWidget:
        panel = self._take_control_tab(panel)
        scroll = QScrollArea()
        scroll.setObjectName(object_name)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setWidget(panel)
        setattr(self, attribute_name, scroll)
        return scroll

    def _take_control_tab(self: Any, panel: QWidget) -> QWidget:
        if hasattr(self, "control_tabs"):
            index = self.control_tabs.indexOf(panel)
            if index >= 0:
                self.control_tabs.removeTab(index)
            if self.control_tabs.count() == 0:
                self.control_tabs.hide()
                if hasattr(self, "left_controls_scroll"):
                    self.left_controls_scroll.hide()
                    if hasattr(self, "main_splitter"):
                        self.left_controls_scroll.setParent(None)
        return panel

    def _take_run_panel(self: Any) -> QWidget:
        panel = self.run_group
        if hasattr(self, "files_tab") and self.files_tab.layout() is not None:
            self.files_tab.layout().removeWidget(panel)
        return panel

    def _frame_matrix_enabled(self: Any) -> bool:
        if not hasattr(self, "show_frame_matrix_checkbox"):
            return True
        return bool(self.show_frame_matrix_checkbox.isChecked())

    def _frame_matrix_thumbnails_enabled(self: Any) -> bool:
        if not self._frame_matrix_enabled():
            return False
        if not hasattr(self, "show_frame_matrix_thumbnails_checkbox"):
            return True
        return bool(self.show_frame_matrix_thumbnails_checkbox.isChecked())

    def _normalized_path(self: Any, path: str | Path) -> str:
        if isinstance(path, str):
            if path in self._image_path_to_index:
                return path
            path_index = getattr(self, "_thumbnail_path_to_row", None)
            if path_index is not None and path in path_index:
                return path
        return str(Path(path))

    def _sync_frame_matrix_controls(self: Any) -> None:
        enabled = self._frame_matrix_enabled()
        if hasattr(self, "show_frame_matrix_thumbnails_checkbox"):
            self.show_frame_matrix_thumbnails_checkbox.setEnabled(enabled)
        if hasattr(self, "thumbnail_matrix_panel"):
            self.thumbnail_matrix_panel.setVisible(enabled)

    def _disable_frame_matrix_runtime(self: Any) -> None:
        self._thumbnail_grid_build_generation = int(
            getattr(self, "_thumbnail_grid_build_generation", 0)
        ) + 1
        self._pending_thumbnail_rebuild_after_vectors = False
        self._thumbnail_flush_retry_count = 0
        self._thumbnail_rebuild_in_progress = False
        self._thumbnail_path_to_row.clear()
        self._thumbnail_selected_path = None
        self._thumbnail_loaded_generation.clear()
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        self._cancel_thumbnail_loading()
        if hasattr(self, "thumbnail_grid"):
            self.thumbnail_grid.clear()
        if hasattr(self, "thumbnail_matrix_panel"):
            self.thumbnail_matrix_panel.hide()

    def _on_frame_matrix_display_settings_changed(self: Any, *_args) -> None:
        self._sync_frame_matrix_controls()
        if self._frame_matrix_enabled():
            self._schedule_thumbnail_grid_rebuild(force=True)
        else:
            self._disable_frame_matrix_runtime()
        self._save_persisted_display_settings()
        self._schedule_frame_prefetch()

    def _on_frame_matrix_thumbnail_settings_changed(self: Any, *_args) -> None:
        thumbnails_enabled = self._frame_matrix_thumbnails_enabled()
        if not thumbnails_enabled:
            self._cancel_thumbnail_loading()
        self._thumbnail_loaded_generation.clear()
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        if hasattr(self, "thumbnail_grid"):
            placeholder = self._thumbnail_placeholder()
            for index in range(self.thumbnail_grid.count()):
                item = self.thumbnail_grid.item(index)
                if item is not None:
                    item.setIcon(placeholder)
            if hasattr(self.thumbnail_grid, "refreshItems"):
                self.thumbnail_grid.refreshItems()
        if thumbnails_enabled:
            self._resume_frame_matrix_thumbnail_loading()
        self._save_persisted_display_settings()

    def _on_thumbnail_viewport_changed(self: Any, *_args) -> None:
        self._schedule_visible_thumbnail_loads_debounced()
        self._schedule_thumbnail_scroll_settle()

    def _schedule_visible_thumbnail_loads_debounced(self: Any) -> None:
        timer = getattr(self, "_thumbnail_visible_load_timer", None)
        if timer is None:
            self._schedule_visible_thumbnail_loads()
            return
        timer.stop()
        timer.start(max(16, int(THUMBNAIL_VISIBLE_LOAD_DEBOUNCE_MS)))
