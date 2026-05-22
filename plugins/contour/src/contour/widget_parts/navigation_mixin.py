from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetNavigationMixin:
    def _dialog_start_directory_from_value(self, value: str | Path | None, fallback: str | Path | None = None) -> str:
        candidates = [value, fallback, Path.home()]
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip().strip("\"'")
            if not text:
                continue
            path = Path(text).expanduser()
            if path.is_dir():
                return str(path)
            if path.is_file() and path.parent.is_dir():
                return str(path.parent)
            if path.parent.is_dir():
                return str(path.parent)
        return str(Path.home())

    def _dialog_start_directory_from_line_edit(self, line_edit, fallback: str | Path | None = None) -> str:
        text = line_edit.text() if hasattr(line_edit, "text") else ""
        return self._dialog_start_directory_from_value(text, fallback)

    def _on_sidebar_list_mode_changed(self, index: int) -> None:
        if index < 0:
            return
        if hasattr(self, "sidebar_list_stack"):
            self.sidebar_list_stack.setCurrentIndex(0 if index == 0 else 1)
        if index == 1:
            # Defer arming: the mouse release that closes the combo popup is often
            # delivered to the vector list in the *next* event-loop tick.
            def _arm_suppress() -> None:
                self._vectors_list_ignore_navigate_until = time.monotonic() + 0.55

            QTimer.singleShot(0, _arm_suppress)
        else:
            self._vectors_list_ignore_navigate_until = 0.0

    def _image_path_for_cif_stem(self, stem: str) -> str | None:
        target = stem.lower()
        for path in self._workspace.image_paths:
            if Path(path).stem.lower() == target:
                return str(Path(path))
        return None

    def _paint_vector_list_item(self, item: QListWidgetItem, stem: str) -> None:
        stem_lower = stem.lower()
        status = self._vector_status_enum_for_stem(stem_lower)
        paint_vector_row_item(item, stem, status, theme=getattr(self, "_ui_theme", "dark"))

    def _vector_status_enum_for_stem(self, stem_lower: str):
        ipath = self._image_path_for_cif_stem(stem_lower)
        has_matching = ipath is not None
        cif_failed = stem_lower in self._cif_load_failure_stems
        normalized = "" if ipath is None else str(Path(ipath))
        never_opened = (not normalized) or (normalized not in self._viewed_image_paths)
        dirty = bool(ipath is not None and self._workspace.image_has_changes(normalized))
        persist = normalized in self._persisted_highlight_paths if normalized else False
        return classify_vector_side_status(
            has_matching_image=has_matching,
            cif_load_failed=cif_failed,
            image_never_viewed=never_opened,
            polygons_dirty=dirty,
            persist_highlight=persist,
        )

    def _rebuild_vector_list(self) -> None:
        if not hasattr(self, "vector_list"):
            return
        self.vector_list.blockSignals(True)
        self.vector_list.clear()
        mapping = sorted(self._workspace.cif_paths_by_stem.items(), key=lambda kv: kv[0].lower())
        if len(mapping) > LARGE_FRAME_COUNT_THRESHOLD:
            count = len(mapping)
            summary = (
                f"Проиндексировано {count} векторов — используйте навигацию по кадрам."
                if self._ui_language == "ru"
                else f"{count} vectors indexed — use frame navigation."
            )
            item = QListWidgetItem(summary)
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.vector_list.addItem(item)
        else:
            for stem, cif_path in mapping:
                item = QListWidgetItem(Path(cif_path).stem)
                item.setToolTip(cif_path)
                item.setData(Qt.ItemDataRole.UserRole, cif_path)
                self._paint_vector_list_item(item, stem)
                self.vector_list.addItem(item)
        self.vector_list.blockSignals(False)
        if len(self._workspace.image_paths) <= ASSET_FILTER_LISTS_MAX_FRAMES:
            self._rebuild_asset_filter_lists()
            self._apply_asset_view_filter()

    def _vector_index_active(self) -> bool:
        return bool(getattr(self._workspace, "_cif_paths_by_stem", {}))

    def _image_path_has_matching_vector(self, image_path: str) -> bool:
        return bool(self._workspace.resolve_cif_path(image_path))

    def _make_asset_image_item(self, image_path: str, *, show_text: bool = True) -> QListWidgetItem:
        item = QListWidgetItem(Path(image_path).stem)
        item.setToolTip(f"Путь к файлу: {image_path}" if self._ui_language == "ru" else f"File path: {image_path}")
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        self._paint_image_row_item(item, image_path, show_text=show_text)
        return item

    def _make_asset_vector_item(self, stem: str, vector_path: str) -> QListWidgetItem:
        item = QListWidgetItem(Path(vector_path).stem)
        item.setToolTip(vector_path)
        item.setData(Qt.ItemDataRole.UserRole, vector_path)
        self._paint_vector_list_item(item, stem)
        return item

    def _update_asset_items_for_image_path(self, image_path: str) -> None:
        if not hasattr(self, "image_vector_list"):
            return
        normalized = str(Path(image_path))
        match_only = bool(getattr(self, "_asset_filter_match_only", False))
        has_vector = self._image_path_has_matching_vector(normalized)
        for list_widget in (self.image_vector_list, self.image_only_list):
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if item is None:
                    continue
                if str(item.data(Qt.ItemDataRole.UserRole) or "") != normalized:
                    continue
                self._paint_image_row_item(item, normalized)
                item.setHidden(bool(match_only and not has_vector))
                return

    def _rebuild_asset_filter_lists(self) -> None:
        if not hasattr(self, "image_vector_list"):
            return
        if len(self._workspace.image_paths) > ASSET_FILTER_LISTS_MAX_FRAMES:
            return
        self._asset_list_build_generation += 1
        generation = self._asset_list_build_generation
        image_paths = [str(Path(path)) for path in self._workspace.image_paths]
        vector_items = sorted(self._workspace.cif_paths_by_stem.items(), key=lambda kv: kv[0].lower())
        sets = build_frame_asset_sets(image_paths, self._workspace.cif_paths_by_stem)
        list_widgets = (self.image_vector_list, self.image_only_list, self.vector_only_list)
        for list_widget in list_widgets:
            list_widget.clear()

        chunk_size = max(1, int(getattr(self, "_image_list_build_chunk_size", 250)))

        def _set_updates_enabled(enabled: bool) -> None:
            for widget in list_widgets:
                widget.setUpdatesEnabled(enabled)

        def _add_image_chunk(start: int) -> None:
            if generation != self._asset_list_build_generation:
                return
            end = min(len(image_paths), start + chunk_size)
            for list_widget in list_widgets:
                list_widget.blockSignals(True)
            _set_updates_enabled(False)
            try:
                for image_path in image_paths[start:end]:
                    stem = Path(image_path).stem.lower()
                    if stem in sets.image_and_vector_stems:
                        self.image_vector_list.addItem(self._make_asset_image_item(image_path))
                    elif stem in sets.image_only_stems:
                        self.image_only_list.addItem(self._make_asset_image_item(image_path))
            finally:
                _set_updates_enabled(True)
                for list_widget in list_widgets:
                    list_widget.blockSignals(False)
            if end < len(image_paths):
                QTimer.singleShot(0, lambda next_start=end: _add_image_chunk(next_start))
                return
            _add_vector_chunk(0)

        def _add_vector_chunk(start: int) -> None:
            if generation != self._asset_list_build_generation:
                return
            end = min(len(vector_items), start + chunk_size)
            self.vector_only_list.blockSignals(True)
            self.vector_only_list.setUpdatesEnabled(False)
            try:
                for stem, vector_path in vector_items[start:end]:
                    if stem.lower() in sets.vector_only_stems:
                        self.vector_only_list.addItem(self._make_asset_vector_item(stem, vector_path))
            finally:
                self.vector_only_list.setUpdatesEnabled(True)
                self.vector_only_list.blockSignals(False)
            if end < len(vector_items):
                QTimer.singleShot(0, lambda next_start=end: _add_vector_chunk(next_start))

        QTimer.singleShot(0, lambda: _add_image_chunk(0))

    def _apply_asset_view_filter(self) -> None:
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

    def _on_asset_image_item_clicked(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        self._set_image_list_current_path(path, fallback_to_first=False)

    def _on_asset_vector_item_clicked(self, item: QListWidgetItem) -> None:
        self._on_vector_item_navigate_request(item)

    def _take_thumbnail_matrix_panel(self) -> QWidget:
        panel = self.thumbnail_matrix_panel
        if hasattr(self, "files_tab") and self.files_tab.layout() is not None:
            self.files_tab.layout().removeWidget(panel)
        return panel

    def _take_files_panel(self) -> QWidget:
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

    def _take_paths_panel(self) -> QWidget:
        return self._take_control_tab(self.paths_tab)

    def _take_pipeline_panel(self) -> QWidget:
        return self._take_control_tab(self.pipeline_tab)

    def _take_display_panel(self) -> QWidget:
        return self._take_control_tab(self.display_tab)

    def _take_recognition_panel(self) -> QWidget:
        return self._take_control_tab(self.extraction_tab)

    def _take_control_tab(self, panel: QWidget) -> QWidget:
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

    def _take_run_panel(self) -> QWidget:
        panel = self.run_group
        if hasattr(self, "files_tab") and self.files_tab.layout() is not None:
            self.files_tab.layout().removeWidget(panel)
        return panel

    def _frame_matrix_enabled(self) -> bool:
        if not hasattr(self, "show_frame_matrix_checkbox"):
            return True
        return bool(self.show_frame_matrix_checkbox.isChecked())

    def _frame_matrix_thumbnails_enabled(self) -> bool:
        if not self._frame_matrix_enabled():
            return False
        if not hasattr(self, "show_frame_matrix_thumbnails_checkbox"):
            return True
        return bool(self.show_frame_matrix_thumbnails_checkbox.isChecked())

    def _sync_frame_matrix_controls(self) -> None:
        enabled = self._frame_matrix_enabled()
        if hasattr(self, "show_frame_matrix_thumbnails_checkbox"):
            self.show_frame_matrix_thumbnails_checkbox.setEnabled(enabled)
        if hasattr(self, "thumbnail_matrix_panel"):
            self.thumbnail_matrix_panel.setVisible(enabled)

    def _disable_frame_matrix_runtime(self) -> None:
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

    def _on_frame_matrix_display_settings_changed(self, *_args) -> None:
        self._sync_frame_matrix_controls()
        if self._frame_matrix_enabled():
            self._schedule_thumbnail_grid_rebuild(force=True)
        else:
            self._disable_frame_matrix_runtime()
        self._save_persisted_display_settings()

    def _on_frame_matrix_thumbnail_settings_changed(self, *_args) -> None:
        if not self._frame_matrix_thumbnails_enabled():
            self._cancel_thumbnail_loading()
        self._thumbnail_loaded_generation.clear()
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        if hasattr(self, "thumbnail_grid"):
                placeholder = self._thumbnail_placeholder()
                for index in range(self.thumbnail_grid.count()):
                    item = self.thumbnail_grid.item(index)
                    if item is not None:
                        item.setIcon(placeholder)
        else:
            self._schedule_visible_thumbnail_loads()
        self._save_persisted_display_settings()

    def _on_thumbnail_viewport_changed(self, *_args) -> None:
        self._schedule_visible_thumbnail_loads()

    def _configure_thumbnail_grid_geometry(self) -> None:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        columns = self._thumbnail_columns()
        icon_size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        cell_w = int(icon_size.width())
        cell_h = int(icon_size.height())
        self.thumbnail_grid.setIconSize(icon_size)
        self.thumbnail_grid.setGridSize(QSize(cell_w, cell_h))
        self.thumbnail_grid.setSpacing(0)
        frame = 2 * int(self.thumbnail_grid.frameWidth())
        item_count = max(1, self._thumbnail_layout_slot_count())
        rows = max(1, int(np.ceil(item_count / float(columns))))
        # QListView icon wrap treats width==columns*cell_w as one column too few.
        grid_width = columns * cell_w + 1 + frame
        grid_height = max(cell_h + frame, rows * cell_h + frame)
        self.thumbnail_grid.setFixedSize(grid_width, grid_height)
        self.thumbnail_grid.doItemsLayout()

    def _thumbnail_layout_slot_count(self) -> int:
        if not hasattr(self, "thumbnail_grid"):
            return 0
        if not bool(getattr(self, "_asset_filter_match_only", False)):
            return self.thumbnail_grid.count()
        highest_visible_index = -1
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is not None and not item.isHidden():
                highest_visible_index = index
        return highest_visible_index + 1

    def _thumbnail_columns(self) -> int:
        if not hasattr(self, "neighbor_columns_spin"):
            return 3
        try:
            return max(1, int(self.neighbor_columns_spin.value()))
        except (TypeError, ValueError):
            return 1

    def _thumbnail_placeholder(self) -> QIcon:
        if not getattr(self, "_thumbnail_placeholder_icon", QIcon()).isNull():
            return self._thumbnail_placeholder_icon
        size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        pixmap = QPixmap(size)
        pixmap.fill(QColor("#1F2937"))
        self._thumbnail_placeholder_icon = QIcon(pixmap)
        return self._thumbnail_placeholder_icon

    def _thumbnail_paths_for_matrix(self) -> list[str]:
        if not self._frame_matrix_enabled():
            self._thumbnail_sparse_window_start = 0
            return []
        all_paths = [str(Path(path)) for path in self._workspace.image_paths]
        self._thumbnail_sparse_window_start = 0
        return all_paths

    def _rebuild_thumbnail_grid(self) -> None:
        if not self._frame_matrix_enabled():
            self._disable_frame_matrix_runtime()
            return
        if not hasattr(self, "thumbnail_grid"):
            return
        self._cancel_thumbnail_loading()
        generation = self._thumbnail_generation
        self._thumbnail_loaded_generation.clear()
        self._thumbnail_rebuild_in_progress = True
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        self._thumbnail_selected_path = None
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass
        self.thumbnail_grid.blockSignals(True)
        self.thumbnail_grid.setUpdatesEnabled(False)
        try:
            self.thumbnail_grid.clear()
        finally:
            self.thumbnail_grid.setUpdatesEnabled(True)
            self.thumbnail_grid.blockSignals(False)
        paths = self._thumbnail_paths_for_matrix()
        chunk_size = max(1, int(getattr(self, "_thumbnail_build_chunk_size", 50)))
        default_interval = 25
        if self._uses_large_frame_list():
            default_interval = 8
        chunk_interval_ms = max(1, int(getattr(self, "_thumbnail_build_interval_ms", default_interval)))
        if len(paths) <= chunk_size:
            self.thumbnail_grid.blockSignals(True)
            self.thumbnail_grid.setUpdatesEnabled(False)
            try:
                for path in paths:
                    self.thumbnail_grid.addItem(self._make_thumbnail_grid_item(path))
            finally:
                self.thumbnail_grid.setUpdatesEnabled(True)
                self.thumbnail_grid.blockSignals(False)
            self._finish_thumbnail_grid_build()
            return

        def _add_thumbnail_chunk(start: int) -> None:
            if generation != self._thumbnail_generation or not hasattr(self, "thumbnail_grid"):
                return
            end = min(len(paths), start + chunk_size)
            self.thumbnail_grid.blockSignals(True)
            self.thumbnail_grid.setUpdatesEnabled(False)
            try:
                for path in paths[start:end]:
                    self.thumbnail_grid.addItem(self._make_thumbnail_grid_item(path))
            finally:
                self.thumbnail_grid.setUpdatesEnabled(True)
                self.thumbnail_grid.blockSignals(False)
            if end < len(paths):
                QTimer.singleShot(chunk_interval_ms, lambda next_start=end: _add_thumbnail_chunk(next_start))
                return
            self._finish_thumbnail_grid_build()

        QTimer.singleShot(chunk_interval_ms, lambda: _add_thumbnail_chunk(0))

    def _finish_thumbnail_grid_build(self) -> None:
        self._thumbnail_rebuild_in_progress = False
        self._configure_thumbnail_grid_geometry()
        self._rebuild_thumbnail_path_index()
        self._apply_asset_view_filter()
        self._update_thumbnail_grid_selection()
        self._schedule_visible_thumbnail_loads()

    def _rebuild_thumbnail_path_index(self) -> None:
        mapping: dict[str, int] = {}
        if hasattr(self, "thumbnail_grid"):
            for index in range(self.thumbnail_grid.count()):
                item = self.thumbnail_grid.item(index)
                if item is None:
                    continue
                path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if path:
                    mapping[path] = index
        self._thumbnail_path_to_row = mapping

    def _schedule_thumbnail_sparse_recenter(self) -> None:
        self._schedule_visible_thumbnail_loads()

    def _ensure_thumbnail_sparse_window_covers_current(self) -> None:
        self._schedule_thumbnail_sparse_recenter()

    def _make_thumbnail_grid_item(self, path: str) -> QListWidgetItem:
        item = QListWidgetItem(self._thumbnail_placeholder(), "")
        item.setSizeHint(self._thumbnail_icon_size)
        item.setToolTip(Path(str(path)).stem)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        if bool(getattr(self, "_asset_filter_match_only", False)) and not self._image_path_has_matching_vector(str(path)):
            item.setHidden(True)
        return item

    def _thumbnail_loading_blocked(self) -> bool:
        if not self._frame_matrix_enabled() or not self._frame_matrix_thumbnails_enabled():
            return True
        if bool(getattr(self, "_thumbnail_rebuild_in_progress", False)):
            return True
        if getattr(self, "_frame_load_running_path", None) is not None:
            return True
        if getattr(self, "_loading_image_path", None) is not None:
            return True
        return False

    def _cancel_thumbnail_loading(self) -> None:
        self._thumbnail_generation += 1
        self._thumbnail_queued_paths.clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()
        self._thumbnail_radial_paths = []
        self._thumbnail_radial_cursor = 0
        self._thumbnail_radial_center_path = None
        if hasattr(self, "_thumbnail_radial_pump_timer"):
            self._thumbnail_radial_pump_timer.stop()
        if hasattr(self, "_thumbnail_apply_timer"):
            self._thumbnail_apply_timer.stop()
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass

    def _pause_thumbnail_radial_fill(self) -> None:
        self._thumbnail_generation += 1
        if hasattr(self, "_thumbnail_radial_pump_timer"):
            self._thumbnail_radial_pump_timer.stop()
        if hasattr(self, "_thumbnail_apply_timer"):
            self._thumbnail_apply_timer.stop()
        self._thumbnail_queued_paths.clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass

    def _thumbnail_grid_paths_in_radial_order(self, center_row: int) -> list[str]:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return []
        columns = max(1, self._thumbnail_columns())
        center_r = center_row // columns
        center_c = center_row % columns
        ordered: list[tuple[int, int, str]] = []
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is None or item.isHidden():
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if not path:
                continue
            row = index // columns
            col = index % columns
            ring = max(abs(row - center_r), abs(col - center_c))
            ordered.append((ring, index, str(Path(path))))
        ordered.sort(key=lambda entry: (entry[0], entry[1]))
        return [path for _, _, path in ordered]

    def _reseed_thumbnail_radial_fill(self) -> None:
        if not self._frame_matrix_thumbnails_enabled():
            return
        if not hasattr(self, "thumbnail_grid") or self.thumbnail_grid.count() <= 0:
            return
        if self._thumbnail_loading_blocked():
            return
        current = self._workspace.current_image_path
        if not current:
            return
        normalized_current = str(Path(current))
        path_index = getattr(self, "_thumbnail_path_to_row", {})
        center_row = path_index.get(normalized_current)
        if center_row is None:
            return
        self._thumbnail_generation += 1
        self._thumbnail_queued_paths.clear()
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass
        loaded_generation = getattr(self, "_thumbnail_loaded_generation", {})
        current_generation = self._thumbnail_generation
        self._thumbnail_radial_paths = [
            path
            for path in self._thumbnail_grid_paths_in_radial_order(center_row)
            if loaded_generation.get(path) != current_generation
        ]
        self._thumbnail_radial_cursor = 0
        self._thumbnail_radial_center_path = normalized_current
        self._schedule_thumbnail_radial_pump()

    def _resume_thumbnail_radial_fill(self) -> None:
        if not self._frame_matrix_thumbnails_enabled():
            return
        if self._thumbnail_loading_blocked():
            return
        current = self._workspace.current_image_path
        if not current:
            return
        normalized = str(Path(current))
        if normalized != getattr(self, "_thumbnail_radial_center_path", None):
            self._reseed_thumbnail_radial_fill()
            return
        paths = getattr(self, "_thumbnail_radial_paths", [])
        cursor = int(getattr(self, "_thumbnail_radial_cursor", 0))
        if cursor < len(paths):
            self._schedule_thumbnail_radial_pump()
            return
        self._reseed_thumbnail_radial_fill()

    def _schedule_thumbnail_radial_pump(self) -> None:
        if not hasattr(self, "_thumbnail_radial_pump_timer"):
            return
        if self._thumbnail_loading_blocked():
            return
        if not getattr(self, "_thumbnail_radial_paths", []):
            return
        self._thumbnail_radial_pump_timer.stop()
        self._thumbnail_radial_pump_timer.start(max(16, int(THUMBNAIL_RADIAL_PUMP_INTERVAL_MS)))

    def _pump_thumbnail_radial_loads(self) -> None:
        if self._thumbnail_loading_blocked():
            return
        paths: list[str] = getattr(self, "_thumbnail_radial_paths", [])
        cursor = int(getattr(self, "_thumbnail_radial_cursor", 0))
        if cursor >= len(paths):
            return
        generation = self._thumbnail_generation
        per_pump = max(1, int(THUMBNAIL_RADIAL_LOADS_PER_PUMP))
        queued = 0
        while cursor < len(paths) and queued < per_pump:
            path = paths[cursor]
            cursor += 1
            before = len(self._thumbnail_queued_paths)
            self._queue_thumbnail_load(generation, path)
            if len(self._thumbnail_queued_paths) > before:
                queued += 1
        self._thumbnail_radial_cursor = cursor
        if cursor < len(paths):
            self._schedule_thumbnail_radial_pump()

    def _schedule_thumbnail_icon_apply(self) -> None:
        if not hasattr(self, "_thumbnail_apply_timer"):
            return
        self._thumbnail_apply_timer.stop()
        self._thumbnail_apply_timer.start(16)

    def _flush_thumbnail_icon_batch(self) -> None:
        if not self._frame_matrix_thumbnails_enabled() or not hasattr(self, "thumbnail_grid"):
            getattr(self, "_thumbnail_pending_apply", {}).clear()
            return
        if bool(getattr(self, "_thumbnail_rebuild_in_progress", False)):
            self._schedule_thumbnail_icon_apply()
            return
        pending: dict[str, object] = getattr(self, "_thumbnail_pending_apply", {})
        if not pending:
            return
        per_tick = max(1, int(THUMBNAIL_ICONS_APPLY_PER_TICK))
        batch_paths = list(pending.keys())[:per_tick]
        placeholder = self._thumbnail_placeholder()
        icon_cache = getattr(self, "_thumbnail_icon_cache", {})
        self.thumbnail_grid.setUpdatesEnabled(False)
        try:
            for path in batch_paths:
                qimage = pending.pop(path, None)
                row = getattr(self, "_thumbnail_path_to_row", {}).get(path)
                if row is None:
                    continue
                item = self.thumbnail_grid.item(row)
                if item is None:
                    continue
                icon = icon_cache.get(path)
                if icon is None:
                    if qimage is None or (hasattr(qimage, "isNull") and qimage.isNull()):
                        icon = placeholder
                    else:
                        pixmap = QPixmap.fromImage(qimage)
                        icon = QIcon(pixmap) if not pixmap.isNull() else placeholder
                    icon_cache[path] = icon
                item.setIcon(icon)
        finally:
            self.thumbnail_grid.setUpdatesEnabled(True)
        if pending:
            self._schedule_thumbnail_icon_apply()

    def _queue_thumbnail_load(self, generation: int, path: str) -> None:
        if self._thumbnail_loading_blocked():
            return
        normalized = str(Path(path))
        if (
            getattr(self, "_thumbnail_loaded_generation", {}).get(normalized) == generation
            or normalized in self._thumbnail_queued_paths
        ):
            return
        self._thumbnail_queued_paths.add(normalized)
        runnable = ThumbnailLoadRunnable(
            generation,
            normalized,
            self._thumbnail_icon_size.width(),
            self._thumbnail_icon_size.height(),
        )
        runnable.signals.result.connect(self._on_thumbnail_loaded)
        self._thumbnail_thread_pool.start(runnable)

    def _visible_thumbnail_rows(self, *, buffer_rows: int = 2) -> tuple[int, int]:
        if not hasattr(self, "thumbnail_grid_scroll_area"):
            return (0, -1)
        icon_size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        cell_h = max(1, int(icon_size.height()))
        vertical = self.thumbnail_grid_scroll_area.verticalScrollBar()
        viewport_h = max(1, int(self.thumbnail_grid_scroll_area.viewport().height()))
        first_row = max(0, (int(vertical.value()) // cell_h) - buffer_rows)
        last_row = max(first_row, ((int(vertical.value()) + viewport_h) // cell_h) + buffer_rows)
        return first_row, last_row

    def _visible_thumbnail_indexes(self) -> list[int]:
        if not self._frame_matrix_thumbnails_enabled() or not hasattr(self, "thumbnail_grid"):
            return []
        columns = max(1, self._thumbnail_columns())
        first_row, last_row = self._visible_thumbnail_rows()
        count = self.thumbnail_grid.count()
        indexes: set[int] = set()
        for row in range(first_row, last_row + 1):
            start = row * columns
            end = min(count, start + columns)
            indexes.update(range(start, end))
        current = self._workspace.current_image_path
        if current:
            center = getattr(self, "_thumbnail_path_to_row", {}).get(str(Path(current)))
            if center is not None:
                current_row = center // columns
                for row in range(max(0, current_row - 2), current_row + 3):
                    start = row * columns
                    end = min(count, start + columns)
                    indexes.update(range(start, end))
        return sorted(index for index in indexes if 0 <= index < count)

    def _schedule_visible_thumbnail_loads(self) -> None:
        if self._thumbnail_loading_blocked() or not hasattr(self, "thumbnail_grid"):
            return
        generation = self._thumbnail_generation
        queued_paths: list[str] = []
        for index in self._visible_thumbnail_indexes():
            item = self.thumbnail_grid.item(index)
            if item is None or item.isHidden():
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if path:
                queued_paths.append(str(Path(path)))
        keep = set(queued_paths)
        current = self._workspace.current_image_path
        if current:
            keep.add(str(Path(current)))
        icon_cache = getattr(self, "_thumbnail_icon_cache", {})
        if len(icon_cache) > 256:
            for path in list(icon_cache.keys()):
                if path not in keep:
                    icon_cache.pop(path, None)
        for path in queued_paths:
            self._queue_thumbnail_load(generation, path)

    def _on_thumbnail_loaded(self, generation: int, path: str, qimage: object) -> None:
        normalized_path = str(Path(path))
        self._thumbnail_queued_paths.discard(normalized_path)
        if generation != self._thumbnail_generation or not hasattr(self, "thumbnail_grid"):
            return
        if not self._frame_matrix_thumbnails_enabled():
            return
        getattr(self, "_thumbnail_loaded_generation", {})[normalized_path] = generation
        pending = getattr(self, "_thumbnail_pending_apply", {})
        pending[normalized_path] = qimage
        getattr(self, "_thumbnail_icon_cache", {}).pop(normalized_path, None)
        self._schedule_thumbnail_icon_apply()

    def _prime_visible_thumbnail_reload(self) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        loaded_generation = getattr(self, "_thumbnail_loaded_generation", {})
        icon_cache = getattr(self, "_thumbnail_icon_cache", {})
        paths: set[str] = set()
        for index in self._visible_thumbnail_indexes():
            item = self.thumbnail_grid.item(index)
            if item is None or item.isHidden():
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if path:
                paths.add(str(Path(path)))
        current = self._workspace.current_image_path
        if current:
            paths.add(str(Path(current)))
        for path in paths:
            loaded_generation.pop(path, None)
            icon_cache.pop(path, None)

    def _resume_frame_matrix_thumbnail_loading(self) -> None:
        if not self._frame_matrix_thumbnails_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        if self._thumbnail_loading_blocked():
            return
        self._prime_visible_thumbnail_reload()
        self._schedule_thumbnail_icon_apply()
        QTimer.singleShot(0, self._schedule_visible_thumbnail_loads)
        QTimer.singleShot(0, self._reseed_thumbnail_radial_fill)

    def _update_thumbnail_grid_selection(self) -> None:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        current = self._workspace.current_image_path
        previous = getattr(self, "_thumbnail_selected_path", None)
        path_index = getattr(self, "_thumbnail_path_to_row", {})
        matched_index = -1
        previous_item = None
        current_item = None
        if previous:
            previous_row = path_index.get(str(Path(previous)))
            if previous_row is not None:
                previous_item = self.thumbnail_grid.item(previous_row)
        if current:
            matched_index = path_index.get(str(Path(current)), -1)
            if matched_index >= 0:
                current_item = self.thumbnail_grid.item(matched_index)
        self.thumbnail_grid.blockSignals(True)
        try:
            if previous_item is not None:
                previous_item.setBackground(QBrush())
                previous_item.setData(Qt.ItemDataRole.BackgroundRole, None)
            if current_item is not None and matched_index >= 0 and not current_item.isHidden():
                self.thumbnail_grid.setCurrentRow(matched_index)
                current_item.setBackground(QBrush(QColor("#1D4ED8")))
                current_item.setData(Qt.ItemDataRole.BackgroundRole, QColor("#1D4ED8"))
                self._thumbnail_selected_path = str(current)
            else:
                self.thumbnail_grid.clearSelection()
                self.thumbnail_grid.setCurrentRow(-1)
                self._thumbnail_selected_path = None
        finally:
            self.thumbnail_grid.blockSignals(False)
        if matched_index >= 0:
            self._scroll_thumbnail_grid_to_row(matched_index)
        current_path = str(Path(current)) if current else None
        if current_path and current_path != getattr(self, "_thumbnail_radial_center_path", None):
            if not self._thumbnail_loading_blocked():
                self._schedule_visible_thumbnail_loads()

    def _scroll_thumbnail_grid_to_row(self, row: int) -> None:
        if not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid") or not hasattr(self, "thumbnail_grid_scroll_area"):
            return
        if row < 0 or row >= self.thumbnail_grid.count():
            return
        item = self.thumbnail_grid.item(row)
        if item is None:
            return
        rect = self.thumbnail_grid.visualItemRect(item)
        if rect.isNull():
            return
        self.thumbnail_grid_scroll_area.ensureVisible(rect.center().x(), rect.center().y(), 0, 0)
        viewport = self.thumbnail_grid_scroll_area.viewport().rect()
        horizontal = self.thumbnail_grid_scroll_area.horizontalScrollBar()
        vertical = self.thumbnail_grid_scroll_area.verticalScrollBar()
        margin = 0
        if rect.left() < horizontal.value():
            horizontal.setValue(max(horizontal.minimum(), rect.left() - margin))
        elif rect.right() > horizontal.value() + viewport.width():
            horizontal.setValue(min(horizontal.maximum(), rect.right() - viewport.width() + margin))
        if rect.top() < vertical.value():
            vertical.setValue(max(vertical.minimum(), rect.top() - margin))
        elif rect.bottom() > vertical.value() + viewport.height():
            vertical.setValue(min(vertical.maximum(), rect.bottom() - viewport.height() + margin))
        self._schedule_visible_thumbnail_loads()

    def _on_thumbnail_item_clicked(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        self._set_image_list_current_path(path, fallback_to_first=False)

    def _refresh_vector_rows_for_workspace(self) -> None:
        if not hasattr(self, "vector_list"):
            return
        if len(self._workspace.cif_paths_by_stem) > LARGE_FRAME_COUNT_THRESHOLD:
            return
        for index in range(self.vector_list.count()):
            row = self.vector_list.item(index)
            if row is None:
                continue
            tip = row.toolTip()
            if not tip:
                continue
            self._paint_vector_list_item(row, Path(tip).stem.lower())

    def _refresh_vector_items_for_stems(self, stems: set[str]) -> None:
        if not stems or not hasattr(self, "vector_list"):
            return
        lowered = {s.lower() for s in stems}
        for idx in range(self.vector_list.count()):
            item = self.vector_list.item(idx)
            if item is None:
                continue
            tip = item.toolTip()
            if not tip:
                continue
            stem = Path(tip).stem.lower()
            if stem in lowered:
                self._paint_vector_list_item(item, stem)

    def _select_input_image_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._tr("select_image_files_dialog"),
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
            self._tr("supported_image_files_filter"),
        )
        if not paths:
            return
        self.input_dir_edit.setText(str(Path(paths[0]).parent))
        self._save_persisted_paths()
        self.append_images([str(Path(p)) for p in paths])

    def _merge_cif_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._tr("merge_cif_files_dialog"),
            self._dialog_start_directory_from_line_edit(self.cif_dir_edit),
            self._tr("cif_files_filter"),
        )
        if not paths:
            return
        additions = index_cif_file_paths(paths)
        if not additions:
            return
        self._workspace.merge_cif_paths(additions)
        self.cif_dir_edit.setText(str(Path(paths[0]).parent))
        self._save_persisted_paths()
        self._append_log(self._tr("cif_indexed_log", count=len(self._workspace.cif_paths_by_stem)))
        self._sync_after_cif_index_changed()

    def _sync_after_cif_index_changed(self) -> None:
        self._clear_cif_transient_hints()
        self._refresh_image_list_item_states()
        cur = self._workspace.current_image_path
        if cur:
            state = self._workspace.current_state
            if state is not None and state.source_image is not None:
                try:
                    self._reload_current_frame_vectors()
                except Exception as exc:
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))
            else:
                try:
                    self.load_image(cur, load_vectors=True)
                except Exception as exc:
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))
        if not self._uses_large_frame_list():
            report = self._matching_report()
            self._log_matching_gaps_after_refresh(report)

    def _clear_cif_transient_hints(self) -> None:
        self._cif_load_failure_stems.clear()

    def _invalidate_cif_overlay_for_stems(self, stems: set[str]) -> list[str]:
        if not stems:
            return []
        paths: list[str] = []
        for path in self._workspace.image_paths:
            stem = Path(path).stem.lower()
            if stem in stems:
                self._cif_load_failure_stems.discard(stem)
                paths.append(str(Path(path)))
        if paths:
            self._workspace.invalidate_image_states(paths)
        return paths

    def _reload_cif_overlays_for_selected_vectors(self) -> None:
        stems: set[str] = set()
        for row in self.vector_list.selectedItems():
            tip = row.toolTip()
            if tip:
                stems.add(Path(tip).stem.lower())
        if not stems:
            self._append_log(self._tr("no_vector_selection_for_reload_log"))
            return
        affected = self._invalidate_cif_overlay_for_stems(stems)
        self._finalize_overlay_reload(affected)

    def _reload_cif_overlays_for_selected_images(self) -> None:
        stems: set[str] = {Path(path).stem.lower() for path in self._image_list_selected_paths()}
        cur = self._workspace.current_image_path
        if not stems and cur:
            stems.add(Path(cur).stem.lower())
        if not stems:
            self._append_log(self._tr("no_image_selection_for_reload_log"))
            return
        affected = self._invalidate_cif_overlay_for_stems(stems)
        self._finalize_overlay_reload(affected)

    def _finalize_overlay_reload(self, paths_for_message: list[str]) -> None:
        if not paths_for_message:
            self._append_log(self._tr("cif_reload_no_matching_images_log"))
            return
        unique_stems = sorted({Path(p).stem.lower() for p in paths_for_message})[:16]
        more_txt = ""
        extra = len({Path(p).stem.lower() for p in paths_for_message}) - len(unique_stems)
        if extra > 0:
            more_txt = f" (+{extra})"
        self._append_log(
            self._tr(
                "cif_reload_invalidate_log",
                stems=", ".join(unique_stems),
                more=more_txt,
            )
        )
        self._refresh_image_list_item_states()
        self._refresh_vector_rows_for_workspace()
        cur = self._workspace.current_image_path
        if cur and cur in paths_for_message:
            try:
                self.load_image(cur)
            except Exception as exc:
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _on_vector_item_navigate_request(self, item: QListWidgetItem) -> None:
        """Jump to the matching image frame (Files → Images) after an explicit click."""

        if item is None:
            return
        if time.monotonic() < self._vectors_list_ignore_navigate_until:
            return
        tip = item.toolTip()
        if not tip:
            return
        stem = Path(tip).stem.lower()
        image_path = self._image_path_for_cif_stem(stem)
        if not image_path:
            self._append_log(self._tr("vector_row_no_matching_image_loaded_log"))
            return
        if not self._image_path_in_image_list(image_path):
            return
        if hasattr(self, "sidebar_list_mode_combo"):
            with QSignalBlocker(self.sidebar_list_mode_combo):
                self.sidebar_list_mode_combo.setCurrentIndex(0)
        if hasattr(self, "sidebar_list_stack"):
            self.sidebar_list_stack.setCurrentIndex(0)
        self._set_image_list_current_path(image_path, fallback_to_first=False)

    def _select_input_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_input_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
        )
        if not path:
            return
        directory = self._path_settings.validate_input_directory(path)
        if not directory.available:
            self._append_log(self._tr("input_directory_missing_log", directory=directory.path))
            return
        self.input_dir_edit.setText(directory.path)
        self._save_persisted_paths()
        self._begin_async_directory_scan(directory.path, append=True)

    def _select_cif_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_cif_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.cif_dir_edit),
        )
        if path:
            self.set_cif_directory(path)

    def _select_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_output_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.output_dir_edit),
        )
        if path:
            self.set_output_directory(path)

    def _select_dataset_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_dataset_directory_dialog"),
            self._dialog_start_directory_from_line_edit(self.dataset_dir_edit),
        )
        if path:
            self.set_dataset_directory(path)

    def _apply_input_directory_edit(self) -> None:
        path = self.input_dir_edit.text().strip()
        if path:
            self.set_input_directory(path)
        else:
            self._workspace.replace_image_selection([], is_supported_image=is_image_path)
            self._base_frame_number_by_path = {}
            self._base_frame_numbers = set()
            self._set_image_list_paths([])
            self._rebuild_thumbnail_grid()
            self._clear_extra_layers()
            self._update_extra_layers_enabled_state()
            self._rebuild_vector_list()
            self._refresh_image_list_item_states()
            self._sync_current_state_views()
            self._save_persisted_paths()

    def _apply_cif_directory_edit(self) -> None:
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
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _apply_output_directory_edit(self) -> None:
        self.set_output_directory(self.output_dir_edit.text().strip())

    def _apply_dataset_directory_edit(self) -> None:
        self.set_dataset_directory(self.dataset_dir_edit.text().strip())

    def _choose_external_color(self) -> None:
        self._choose_color("external_color", self.external_color_button)

    def _choose_hole_color(self) -> None:
        self._choose_color("hole_color", self.hole_color_button)

    def _choose_selected_color(self) -> None:
        self._choose_color("selected_color", self.selected_color_button)

    def _choose_conductor_hover_highlight_color(self) -> None:
        self._choose_color("conductor_hover_highlight_color", self.conductor_hover_highlight_color_button)

    def _choose_vertex_color(self) -> None:
        self._choose_color("vertex_color", self.vertex_color_button)

    def _choose_color(self, attribute_name: str, button: QPushButton) -> None:
        initial = QColor(getattr(self._display_settings, attribute_name))
        color = QColorDialog.getColor(initial, self, self._tr("select_color_dialog_title"))
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexRgb)
        setattr(self._display_settings, attribute_name, value)
        self._update_color_button(button, value)
        self._apply_display_settings()

    def _apply_display_settings(self) -> None:
        if hasattr(self, "line_width_spin"):
            self._display_settings.line_width = float(self.line_width_spin.value())
            self._display_settings.vertex_size = float(self.vertex_size_spin.value())
            self._display_settings.fill_opacity = float(self.fill_opacity_spin.value())
            self._display_settings.show_vertices = bool(self.show_vertices_checkbox.isChecked())
            self._display_settings.show_labels = bool(self.show_labels_checkbox.isChecked())
            if hasattr(self, "polygon_editor"):
                self.polygon_editor.set_display_settings(self._display_settings)
                if hasattr(self, "random_object_colors_checkbox"):
                    self.polygon_editor.set_random_object_colors_enabled(self.random_object_colors_checkbox.isChecked())
        self._apply_vector_geometry_editor_config()
        self._save_persisted_display_settings()

    def _on_neighbor_display_settings_changed(self, *_args) -> None:
        self._sync_neighbor_frames()
        self._configure_thumbnail_grid_geometry()
        self._save_persisted_display_settings()

    def _refresh_extra_layers_list(self) -> None:
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

    def _build_extra_layer_row_widget(self, layer: dict[str, object]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layer_id = int(layer.get("id", 0))

        visible_checkbox = QCheckBox("")
        visible_checkbox.setChecked(bool(layer.get("visible", True)))
        visible_checkbox.setToolTip("Показать/скрыть слой")
        visible_checkbox.stateChanged.connect(
            lambda _state, lid=layer_id: self._set_extra_layer_field(lid, "visible", visible_checkbox.isChecked())
        )

        dx_spin = QSpinBox()
        dx_spin.setRange(-1_000_000, 1_000_000)
        dx_spin.setValue(int(layer.get("dx", 0) or 0))
        dx_spin.setMinimumWidth(0)
        dx_spin.setMaximumWidth(self._compact_spinbox_width(dx_spin, "-1000000"))
        dx_spin.setToolTip("Смещение слоя по X")
        dx_spin.valueChanged.connect(lambda value, lid=layer_id: self._set_extra_layer_field(lid, "dx", int(value)))

        dy_spin = QSpinBox()
        dy_spin.setRange(-1_000_000, 1_000_000)
        dy_spin.setValue(int(layer.get("dy", 0) or 0))
        dy_spin.setMinimumWidth(0)
        dy_spin.setMaximumWidth(self._compact_spinbox_width(dy_spin, "-1000000"))
        dy_spin.setToolTip("Смещение слоя по Y")
        dy_spin.valueChanged.connect(lambda value, lid=layer_id: self._set_extra_layer_field(lid, "dy", int(value)))

        opacity_spin = QSpinBox()
        opacity_spin.setRange(0, 100)
        opacity_spin.setValue(int(layer.get("opacity", 100) or 100))
        opacity_spin.setSuffix("%")
        opacity_spin.setMinimumWidth(0)
        opacity_spin.setMaximumWidth(self._compact_spinbox_width(opacity_spin, "100%"))
        opacity_spin.setToolTip("Прозрачность слоя")
        opacity_spin.valueChanged.connect(
            lambda value, lid=layer_id: self._set_extra_layer_field(lid, "opacity", int(value))
        )

        remove_button = QPushButton("-")
        remove_button.setFixedWidth(24)
        remove_button.setStyleSheet("QPushButton { background-color: #DC2626; color: white; font-weight: 700; }")
        remove_button.setToolTip("Удалить слой")
        remove_button.clicked.connect(lambda _checked=False, lid=layer_id: self._remove_extra_layer_by_id(lid))

        folder_name = str(layer.get("name", "Layer"))
        folder_label = QLabel(folder_name)
        folder_label.setToolTip(str(layer.get("folder_path", "")))

        layout.addWidget(visible_checkbox)
        layout.addWidget(folder_label, 1)
        layout.addWidget(dx_spin)
        layout.addWidget(dy_spin)
        layout.addWidget(opacity_spin)
        layout.addWidget(remove_button)
        return row

    @staticmethod
    def _compact_spinbox_width(spinbox: QSpinBox, sample_text: str) -> int:
        text_width = spinbox.fontMetrics().horizontalAdvance(sample_text)
        arrow_width = max(24, spinbox.style().pixelMetric(spinbox.style().PixelMetric.PM_ScrollBarExtent))
        frame = 16
        return text_width + arrow_width + frame

    def _set_extra_layer_field(self, layer_id: int, key: str, value: object) -> None:
        for layer in self._extra_layers:
            if int(layer.get("id", -1)) == layer_id:
                layer[key] = value
                self._sync_extra_layers()
                return

    def _remove_extra_layer_by_id(self, layer_id: int) -> None:
        self._extra_layers = [layer for layer in self._extra_layers if int(layer.get("id", -1)) != layer_id]
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _clear_extra_layers(self) -> None:
        self._extra_layers.clear()
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _update_extra_layers_enabled_state(self) -> None:
        has_base = bool(self._workspace.image_paths)
        if hasattr(self, "add_extra_layers_button"):
            self.add_extra_layers_button.setEnabled(has_base)

    def _sync_extra_layers(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        current_path = self._workspace.current_image_path
        current_number = self._base_frame_number_by_path.get(str(Path(current_path))) if current_path else None
        active_layers: list[dict[str, object]] = []
        if current_number is not None:
            for layer in self._extra_layers:
                if not bool(layer.get("visible", True)):
                    continue
                frame_map = layer.get("frame_map")
                if not isinstance(frame_map, dict):
                    continue
                layer_image_path = frame_map.get(current_number)
                if not isinstance(layer_image_path, str):
                    continue
                pixmap_cache = layer.setdefault("_pixmap_cache", {})
                pixmap = pixmap_cache.get(layer_image_path) if isinstance(pixmap_cache, dict) else None
                if not isinstance(pixmap, QPixmap):
                    pixmap = QPixmap(layer_image_path)
                    if isinstance(pixmap_cache, dict) and not pixmap.isNull():
                        pixmap_cache[layer_image_path] = pixmap
                if pixmap.isNull():
                    continue
                active_layers.append(
                    {
                        "name": layer.get("name", ""),
                        "visible": bool(layer.get("visible", True)),
                        "opacity": max(0.0, min(1.0, float(layer.get("opacity", 100)) / 100.0)),
                        "dx": float(layer.get("dx", 0) or 0),
                        "dy": float(layer.get("dy", 0) or 0),
                        "pixmap": pixmap,
                    }
                )
        self.polygon_editor.set_extra_layers(active_layers)

    def _load_extra_layers(self) -> None:
        if not self._workspace.image_paths:
            QMessageBox.information(self, "Contour", "Сначала загрузите базовый слой")
            return
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку дополнительного слоя"
            if self._ui_language == "ru"
            else "Select additional layer directory",
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
        )
        if not folder_path:
            return
        layer = self._extra_layer_from_directory(folder_path)
        if layer is None:
            return
        self._extra_layers.append(layer)
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _extra_layer_from_directory(self, directory_path: str) -> dict[str, object] | None:
        path = Path(directory_path.strip().strip("\"'")).expanduser()
        if not path.is_dir():
            self._append_log(
                self._tr(
                    "extra_layer_missing_file_log",
                    "Папка слоя не найдена: {path}" if self._ui_language == "ru" else "Layer folder not found: {path}",
                    path=directory_path,
                )
            )
            return None
        layer_image_paths = scan_image_files(path)
        frame_map, warnings = build_additional_layer_frame_map(
            layer_image_paths,
            base_frame_numbers=set(self._base_frame_numbers),
        )
        for message in warnings:
            self._append_log(message)
        if not frame_map:
            self._append_log(
                self._tr(
                    "extra_layer_load_failed_log",
                    "В папке слоя нет кадров, совпадающих с базовым слоем: {path}"
                    if self._ui_language == "ru"
                    else "Layer folder has no frames matching base layer: {path}",
                    path=str(path),
                )
            )
            return None
        layer_id = self._next_extra_layer_id
        self._next_extra_layer_id += 1
        return {
            "id": layer_id,
            "name": path.name,
            "folder_path": str(path),
            "frame_map": frame_map,
            "visible": True,
            "opacity": 100,
            "dx": 0,
            "dy": 0,
        }

    def _on_extra_layers_rows_moved(self, *_args) -> None:
        if not hasattr(self, "extra_layers_list"):
            return
        order: list[int] = []
        for row in range(self.extra_layers_list.count()):
            item = self.extra_layers_list.item(row)
            if item is None:
                continue
            try:
                order.append(int(item.data(Qt.ItemDataRole.UserRole)))
            except (TypeError, ValueError):
                continue
        if not order:
            return
        by_id = {int(layer.get("id", -1)): layer for layer in self._extra_layers}
        self._extra_layers = [by_id[layer_id] for layer_id in order if layer_id in by_id]
        self._sync_extra_layers()


