from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetNavigationMixin:
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
        paint_vector_row_item(item, stem, status)

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
        for stem, cif_path in mapping:
            item = QListWidgetItem(Path(cif_path).stem)
            item.setToolTip(cif_path)
            item.setData(Qt.ItemDataRole.UserRole, cif_path)
            self._paint_vector_list_item(item, stem)
            self.vector_list.addItem(item)
        self.vector_list.blockSignals(False)

    def _configure_thumbnail_grid_geometry(self) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        columns = self._thumbnail_columns()
        icon_size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        cell_w = int(icon_size.width())
        cell_h = int(icon_size.height())
        self.thumbnail_grid.setIconSize(icon_size)
        self.thumbnail_grid.setGridSize(QSize(cell_w, cell_h))
        self.thumbnail_grid.setSpacing(0)
        frame = 2 * int(self.thumbnail_grid.frameWidth())
        item_count = max(1, self.thumbnail_grid.count())
        rows = max(1, int(np.ceil(item_count / float(columns))))
        content_width = max(cell_w + frame, columns * cell_w + frame)
        content_height = max(cell_h + frame, rows * cell_h + frame)
        self.thumbnail_grid.setFixedWidth(content_width)
        self.thumbnail_grid.setFixedHeight(content_height)

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

    def _rebuild_thumbnail_grid(self) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        self._thumbnail_generation += 1
        generation = self._thumbnail_generation
        self._thumbnail_loaded_paths.clear()
        self._thumbnail_queued_paths.clear()
        try:
            self._thumbnail_thread_pool.clear()
        except AttributeError:
            pass
        self._configure_thumbnail_grid_geometry()
        self.thumbnail_grid.blockSignals(True)
        try:
            self.thumbnail_grid.clear()
            for path in self._workspace.image_paths:
                item = QListWidgetItem(self._thumbnail_placeholder(), "")
                item.setSizeHint(self._thumbnail_icon_size)
                item.setToolTip(Path(str(path)).stem)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self._paint_image_row_item(item, str(path), show_text=False)
                self.thumbnail_grid.addItem(item)
        finally:
            self.thumbnail_grid.blockSignals(False)
        self._configure_thumbnail_grid_geometry()
        self._update_thumbnail_grid_selection()

    def _queue_thumbnail_load(self, generation: int, path: str) -> None:
        normalized = str(Path(path))
        if normalized in self._thumbnail_loaded_paths or normalized in self._thumbnail_queued_paths:
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

    def _queue_thumbnail_loads_near_current(self) -> None:
        if not hasattr(self, "thumbnail_grid") or self.thumbnail_grid.count() <= 0:
            return
        generation = self._thumbnail_generation
        current = self._workspace.current_image_path
        if not current:
            return
        current_row = -1
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is not None and current and item.data(Qt.ItemDataRole.UserRole) == current:
                current_row = index
                break
        if current_row < 0:
            return
        columns = self._thumbnail_columns()
        icon_h = max(1, int(self._thumbnail_icon_size.height()))
        viewport_h = icon_h
        if hasattr(self, "thumbnail_grid_scroll_area"):
            viewport_h = max(icon_h, int(self.thumbnail_grid_scroll_area.viewport().height()))
        visible_rows = max(1, int(np.ceil(viewport_h / float(icon_h))))
        radius = max(columns * 4, columns * (visible_rows + 2))
        start = max(0, current_row - radius)
        end = min(self.thumbnail_grid.count(), current_row + radius + 1)
        priority_indexes = [current_row, *range(start, end)]
        seen: set[int] = set()
        for index in priority_indexes:
            if index in seen or index < 0 or index >= self.thumbnail_grid.count():
                continue
            seen.add(index)
            item = self.thumbnail_grid.item(index)
            if item is None:
                continue
            path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if path:
                self._queue_thumbnail_load(generation, path)

    def _on_thumbnail_loaded(self, generation: int, path: str, image: object) -> None:
        normalized_path = str(Path(path))
        self._thumbnail_queued_paths.discard(normalized_path)
        if generation != self._thumbnail_generation or not hasattr(self, "thumbnail_grid"):
            return
        self._thumbnail_loaded_paths.add(normalized_path)
        target = normalized_path
        item = None
        for index in range(self.thumbnail_grid.count()):
            candidate = self.thumbnail_grid.item(index)
            if candidate is not None and str(candidate.data(Qt.ItemDataRole.UserRole) or "") == target:
                item = candidate
                break
        if item is None:
            return
        if image is None:
            item.setIcon(self._thumbnail_placeholder())
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        if pixmap.isNull():
            item.setIcon(self._thumbnail_placeholder())
        else:
            scaled = pixmap.scaled(
                self._thumbnail_icon_size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(QIcon(scaled))

    def _update_thumbnail_grid_selection(self) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        current = self._workspace.current_image_path
        matched = False
        matched_index = -1
        self.thumbnail_grid.blockSignals(True)
        try:
            for index in range(self.thumbnail_grid.count()):
                item = self.thumbnail_grid.item(index)
                selected = bool(current and item is not None and item.data(Qt.ItemDataRole.UserRole) == current)
                if selected:
                    self.thumbnail_grid.setCurrentRow(index)
                    matched = True
                    matched_index = index
                if item is not None:
                    path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                    if selected:
                        item.setBackground(QBrush(QColor("#1D4ED8")))
                        item.setData(Qt.ItemDataRole.BackgroundRole, QColor("#1D4ED8"))
                    elif path:
                        self._paint_image_row_item(item, path, show_text=False)
            if not matched:
                self.thumbnail_grid.clearSelection()
                self.thumbnail_grid.setCurrentRow(-1)
        finally:
            self.thumbnail_grid.blockSignals(False)
        if matched_index >= 0:
            self._scroll_thumbnail_grid_to_row(matched_index)
            QTimer.singleShot(0, lambda row=matched_index: self._scroll_thumbnail_grid_to_row(row))
            self._queue_thumbnail_loads_near_current()

    def _scroll_thumbnail_grid_to_row(self, row: int) -> None:
        if not hasattr(self, "thumbnail_grid") or not hasattr(self, "thumbnail_grid_scroll_area"):
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

    def _on_thumbnail_item_clicked(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        image_item = self._find_image_list_item(path)
        if image_item is not None:
            self.image_list.setCurrentItem(image_item)

    def _refresh_vector_rows_for_workspace(self) -> None:
        if not hasattr(self, "vector_list"):
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
            self.input_dir_edit.text() or str(Path.home()),
            self._tr("supported_image_files_filter"),
        )
        if not paths:
            return
        self.input_dir_edit.setText(str(Path(paths[0]).parent))
        self._save_persisted_paths()
        self.load_images([str(Path(p)) for p in paths])

    def _merge_cif_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._tr("merge_cif_files_dialog"),
            self.cif_dir_edit.text() or str(Path.home()),
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
        self._rebuild_vector_list()
        self._refresh_image_list_item_states()
        cur = self._workspace.current_image_path
        if cur:
            try:
                self.load_image(cur)
            except Exception as exc:
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))
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
        stems: set[str] = {Path(str(row.data(Qt.ItemDataRole.UserRole))).stem.lower() for row in self.image_list.selectedItems()}
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

    def _sync_frame_navigation_controls(self) -> None:
        if not hasattr(self, "frame_nav_spin"):
            return
        paths = list(self._workspace.image_paths)
        total = len(paths)
        self.frame_nav_spin.blockSignals(True)
        if total <= 0:
            self.visual_frame_nav_widget.setEnabled(False)
            self.frame_nav_spin.setMinimum(1)
            self.frame_nav_spin.setMaximum(1)
            self.frame_nav_spin.setValue(1)
            self.frame_nav_total_label.setText("/ 0")
        else:
            self.visual_frame_nav_widget.setEnabled(True)
            self.frame_nav_spin.setMinimum(1)
            self.frame_nav_spin.setMaximum(total)
            current = self._workspace.current_image_path
            position = paths.index(current) + 1 if current and current in paths else min(self.frame_nav_spin.value(), total)
            position = max(1, min(position, total))
            self.frame_nav_spin.setValue(position)
            self.frame_nav_total_label.setText(f"/ {total}")
        self.frame_nav_spin.blockSignals(False)

    def _on_frame_nav_spin_changed(self, value: int) -> None:
        paths = list(self._workspace.image_paths)
        if not paths:
            return
        idx = max(0, min(int(value), len(paths)) - 1)
        target_item = self.image_list.item(idx)
        if target_item is not None:
            self.image_list.setCurrentItem(target_item)

    def _frame_nav_previous(self) -> None:
        if not hasattr(self, "frame_nav_spin"):
            return
        self.frame_nav_spin.setValue(max(1, self.frame_nav_spin.value() - 1))

    def _frame_nav_next(self) -> None:
        if not hasattr(self, "frame_nav_spin"):
            return
        total = len(self._workspace.image_paths)
        if not total:
            return
        self.frame_nav_spin.setValue(min(total, self.frame_nav_spin.value() + 1))

    def _on_image_selection_changed(self) -> None:
        rows = sorted({self.image_list.row(i) for i in self.image_list.selectedItems()})
        paths = list(self._workspace.image_paths)
        if len(rows) == 1 and paths:
            with QSignalBlocker(self.frame_nav_spin):
                self.frame_nav_spin.setValue(min(max(1, rows[0] + 1), len(paths)))

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
        image_item = self._find_image_list_item(image_path)
        if image_item is None:
            return
        if hasattr(self, "sidebar_list_mode_combo"):
            with QSignalBlocker(self.sidebar_list_mode_combo):
                self.sidebar_list_mode_combo.setCurrentIndex(0)
        if hasattr(self, "sidebar_list_stack"):
            self.sidebar_list_stack.setCurrentIndex(0)
        self.image_list.blockSignals(True)
        self.image_list.setCurrentItem(image_item)
        self.image_list.blockSignals(False)

    def _select_input_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_input_directory_dialog"),
            self.input_dir_edit.text(),
        )
        if path:
            self.set_input_directory(path)

    def _select_cif_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_cif_directory_dialog"),
            self.cif_dir_edit.text(),
        )
        if path:
            self.set_cif_directory(path)

    def _select_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_output_directory_dialog"),
            self.output_dir_edit.text(),
        )
        if path:
            self.set_output_directory(path)

    def _select_dataset_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_dataset_directory_dialog"),
            self.dataset_dir_edit.text(),
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
            self.image_list.clear()
            self._rebuild_thumbnail_grid()
            self._clear_extra_layers()
            self._update_extra_layers_enabled_state()
            self._rebuild_vector_list()
            self._refresh_image_list_item_states()
            self._sync_frame_navigation_controls()
            self._sync_current_state_views()
            self._save_persisted_paths()

    def _apply_cif_directory_edit(self) -> None:
        path = self.cif_dir_edit.text().strip()
        if path:
            self.set_cif_directory(path)
        else:
            self._workspace.clear_cif_index()
            self._save_persisted_paths()
            self._rebuild_vector_list()
            self._refresh_image_list_item_states()
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
            "",
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


