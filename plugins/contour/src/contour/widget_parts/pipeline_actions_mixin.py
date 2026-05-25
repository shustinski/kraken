from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetPipelineActionsMixin:
    def _add_pipeline_step(self) -> None:
        operation_name = self._selected_available_operation_name()
        if not operation_name:
            return
        self._pipeline.steps.append(PreprocessingPipeline.create_step(operation_name))
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(len(self._pipeline.steps) - 1)
        self._auto_apply_pipeline()

    def _remove_pipeline_step(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        self._pipeline.steps.pop(row)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def _move_pipeline_step_up(self) -> None:
        row = self.pipeline_list.currentRow()
        if row <= 0:
            return
        self._pipeline.steps[row - 1], self._pipeline.steps[row] = (
            self._pipeline.steps[row],
            self._pipeline.steps[row - 1],
        )
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row - 1)
        self._auto_apply_pipeline()

    def _move_pipeline_step_down(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0 or row >= len(self._pipeline.steps) - 1:
            return
        self._pipeline.steps[row + 1], self._pipeline.steps[row] = (
            self._pipeline.steps[row],
            self._pipeline.steps[row + 1],
        )
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row + 1)
        self._auto_apply_pipeline()

    def _on_pipeline_item_changed(self, item: QListWidgetItem) -> None:
        if self._ignore_pipeline_item_change:
            return
        row = self.pipeline_list.row(item)
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].enabled = item.checkState() == Qt.CheckState.Checked
        self._auto_apply_pipeline()

    def _sync_pipeline_order_from_list(self) -> None:
        if self._ignore_pipeline_item_change:
            return
        old_steps = list(self._pipeline.steps)
        new_steps = []
        for row in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(row)
            old_index = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(old_index, int) or old_index < 0 or old_index >= len(old_steps):
                return
            new_steps.append(old_steps[old_index])
        if len(new_steps) != len(old_steps) or all(
            first is second for first, second in zip(new_steps, old_steps, strict=False)
        ):
            return
        self._pipeline.steps = new_steps
        for row in range(self.pipeline_list.count()):
            self.pipeline_list.item(row).setData(Qt.ItemDataRole.UserRole, row)
        self._render_pipeline_parameters(self.pipeline_list.currentRow())
        self._auto_apply_pipeline()

    def _on_epsilon_spin_value_changed(self, *_args) -> None:
        if hasattr(self, "epsilon_slider"):
            self.epsilon_slider.blockSignals(True)
            try:
                self.epsilon_slider.setValue(min(1000, max(0, round(self.epsilon_spin.value() * 100.0))))
            finally:
                self.epsilon_slider.blockSignals(False)
        self._on_extraction_settings_changed()

    def _on_epsilon_slider_value_changed(self, value: int) -> None:
        self.epsilon_spin.blockSignals(True)
        try:
            self.epsilon_spin.setValue(value / 100.0)
        finally:
            self.epsilon_spin.blockSignals(False)
        self._on_extraction_settings_changed()

    def _on_extraction_settings_changed(self, *_args) -> None:
        # Stop in-flight preview immediately (cooperative cancel); keep prepared-image
        # workers running — pipeline / source unchanged.
        self._abort_in_flight_interactive_processing(preview=True, prepared=False)
        if not hasattr(self, "_extraction_settings_debounce"):
            self._extraction_settings_debounce = QTimer(self)
            self._extraction_settings_debounce.setSingleShot(True)
            self._extraction_settings_debounce.timeout.connect(self._flush_extraction_settings_changed)
        self._extraction_settings_debounce.stop()
        self._extraction_settings_debounce.start(120)

    def _flush_extraction_settings_changed(self) -> None:
        if hasattr(self, "via_white_range_checkbox"):
            self._update_via_threshold_controls_state()
        self._store_active_extraction_profile_settings()
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_debug_candidates([])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
        self._refresh_gradient_overlay()
        self._auto_apply_pipeline()

    def _on_via_search_method_changed(self, *_args) -> None:
        if hasattr(self, "bright_via_mode_stack"):
            self.bright_via_mode_stack.setCurrentIndex(
                1 if self.via_search_mode_combo.currentData() == VIA_SEARCH_MODE_TEMPLATE else 0
            )

    def _sync_via_diameter_size_mode(self, *_args) -> None:
        if hasattr(self, "via_size_mode_combo") and hasattr(self, "via_diameter_size_mode_combo"):
            with QSignalBlocker(self.via_size_mode_combo):
                self.via_size_mode_combo.setCurrentIndex(self.via_diameter_size_mode_combo.currentIndex())
        self._on_via_size_mode_changed()

    def _on_via_size_mode_changed(self, *_args) -> None:
        self._update_via_size_controls_state()
        if (
            normalize_via_size_mode(self.via_size_mode_combo.currentData()) == VIA_SIZE_MODE_FIXED
            and not self._fixed_via_rows
        ):
            self._add_fixed_via_row(width=1, height=1)
            return
        self._on_extraction_settings_changed()

    def _on_extraction_profile_changed(self, *_args) -> None:
        """Legacy hook; profile is controlled by recognition mode."""

    def _store_active_extraction_profile_settings(self) -> None:
        if not hasattr(self, "recognition_mode_combo"):
            return
        rec = str(self.recognition_mode_combo.currentData() or "conductors")
        profile = "vias" if rec == "via" else "conductors"
        self._active_extraction_profile = profile
        settings = self._current_contour_settings()
        # Keep profile snapshots stable: "disabled" is a temporary UI state and
        # must not overwrite the conductors profile recognition mode.
        settings.recognition_mode = "via" if profile == "vias" else "conductors"
        settings.extraction_profile = profile
        settings.object_type = "via" if profile == "vias" else "conductor"
        self._contour_settings_profiles[profile] = settings

    def _save_pipeline_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("save_pipeline_dialog_title"),
            self._dialog_start_directory_from_line_edit(self.output_dir_edit),
            self._tr("json_file_filter"),
        )
        if not path:
            return
        save_pipeline_config_to_path(path, self.get_pipeline())
        self._append_log(self._tr("pipeline_saved_log", path=path))

    def _load_pipeline_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("load_pipeline_dialog_title"),
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
            self._tr("json_file_filter"),
        )
        if not path:
            return
        payload = load_pipeline_config_from_path(path)
        self.set_pipeline(payload)
        self._append_log(self._tr("pipeline_loaded_log", path=path))

    def _on_image_list_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        image_path = self._image_list_path_from_proxy_index(current)
        selection_start = time.perf_counter()
        session = None
        if image_path:
            session = self._start_frame_switch_profile(str(image_path))
            if session is not None:
                self._append_log(
                    f"[contour frame switch profiling] click image={Path(str(image_path)).name} "
                    "(measures selection until UI is interactive)"
                )
        if previous.isValid() and not self._try_leave_current_frame():
            if session is not None:
                session.note_timing("selection_try_leave_cancelled", (time.perf_counter() - selection_start) * 1000.0)
                self._schedule_frame_switch_profile_until_interactive(
                    str(image_path),
                    cif_path=self._find_matching_cif_path(str(image_path)),
                    polygon_count=0,
                    failed=True,
                )
            selection = self.image_list.selectionModel()
            if selection is not None:
                with QSignalBlocker(selection):
                    self.image_list.setCurrentIndex(previous)
            return
        if session is not None:
            session.note_timing("selection_try_leave", (time.perf_counter() - selection_start) * 1000.0)
        if not image_path:
            self._update_thumbnail_grid_selection()
            return
        if image_path:
            try:
                self.load_image(str(image_path))
            except Exception as exc:
                self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=exc))
                QMessageBox.warning(self, self._tr("image_load_error_title"), str(exc))
        self._update_thumbnail_grid_selection()

    def _on_image_item_changed(self, current, previous) -> None:
        image_path = None
        if current is not None:
            value = current.data(Qt.ItemDataRole.UserRole) if hasattr(current, "data") else None
            image_path = str(value) if value else None
        if previous is not None and not self._try_leave_current_frame():
            return
        if not image_path:
            self._update_thumbnail_grid_selection()
            return
        if not self._is_extraction_mode_enabled():
            self._viewed_image_paths.add(str(Path(image_path)))
        try:
            self.load_image(str(image_path))
        except Exception as exc:
            self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=exc))
            QMessageBox.warning(self, self._tr("image_load_error_title"), str(exc))
        for item in (previous, current):
            if item is None:
                continue
            value = item.data(Qt.ItemDataRole.UserRole) if hasattr(item, "data") else None
            if value:
                self._paint_image_row_item(item, str(value))
        self._update_thumbnail_grid_selection()

    def _prune_tagged_sets_for_images(self, retained_paths: list[str]) -> None:
        retained = {str(Path(p)) for p in retained_paths}
        stems = {Path(p).stem.lower() for p in retained_paths}
        self._persisted_highlight_paths.intersection_update(retained)
        self._viewed_image_paths.intersection_update(retained)
        self._cif_load_failure_stems.intersection_update(stems)

    def _matching_report(self):
        image_paths = list(self._workspace.image_paths)
        if len(image_paths) > LARGE_FRAME_COUNT_THRESHOLD:
            return build_image_cif_matching_report(image_paths[:0], self._workspace.cif_paths_by_stem)
        return build_image_cif_matching_report(
            image_paths,
            self._workspace.cif_paths_by_stem,
        )

    def _log_matching_gaps_after_refresh(self, report) -> None:
        if report.stems_with_image_but_no_cif:
            sample = sorted(report.stems_with_image_but_no_cif)[:12]
            more_txt = ""
            extra = len(report.stems_with_image_but_no_cif) - len(sample)
            if extra > 0:
                more_txt = f" (+{extra})"
            self._append_log(
                self._tr(
                    "images_without_matching_cif_log",
                    count=len(report.stems_with_image_but_no_cif),
                    sample=", ".join(sample),
                    more=more_txt,
                )
            )
        if report.stems_with_cif_but_no_image:
            sample = sorted(report.stems_with_cif_but_no_image)[:12]
            more_txt = ""
            extra = len(report.stems_with_cif_but_no_image) - len(sample)
            if extra > 0:
                more_txt = f" (+{extra})"
            self._append_log(
                self._tr(
                    "cif_without_matching_image_log",
                    count=len(report.stems_with_cif_but_no_image),
                    sample=", ".join(sample),
                    more=more_txt,
                )
            )

    def _on_input_directory_scan_started(self, _directory: str) -> None:
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(True)
            self.files_scan_progress_bar.setRange(0, 0)
            self.files_scan_progress_bar.setFormat(self._tr("scanning_directory_progress"))

    def _on_input_directory_scan_idle(self) -> None:
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(False)
            self.files_scan_progress_bar.setRange(0, 100)
            self.files_scan_progress_bar.setValue(0)

    def _begin_async_directory_scan(self, directory: str, *, append: bool = False) -> None:
        self._directory_scan_append_mode = bool(append)
        self._directory_scanner.start(directory)

    def _on_input_directory_scan_finished(self, paths: list[str]) -> None:
        if getattr(self, "_directory_scan_append_mode", False):
            self._directory_scan_append_mode = False
            self.append_images(paths)
            return
        preferred = getattr(self, "_pending_restore_current_image_path", None)
        self._pending_restore_current_image_path = None
        self.load_images(paths, preferred_current_image_path=preferred)

    def _on_input_directory_scan_failed(self, message: str) -> None:
        self._directory_scan_append_mode = False
        self._append_log(
            self._tr(
                "scan_input_directory_failed_log",
                error=message,
            )
        )

    def _on_cif_directory_index_started(self, _directory: str) -> None:
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(True)
            self.files_scan_progress_bar.setRange(0, 0)
            self.files_scan_progress_bar.setFormat(
                "Индексация векторов..." if self._ui_language == "ru" else "Indexing vectors..."
            )

    def _on_cif_directory_index_idle(self) -> None:
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(False)
            self.files_scan_progress_bar.setRange(0, 100)
            self.files_scan_progress_bar.setValue(0)

    def _begin_async_cif_directory_index(self, directory: str) -> None:
        self._vector_indexer.start(directory)

    def _start_vector_index_or_rebuild_frame_matrix_after_images(self) -> None:
        if bool(getattr(self, "_image_list_rebuild_in_progress", False)):
            return
        pending_directory = getattr(self, "_pending_cif_directory_path_after_images", None)
        if pending_directory:
            self._pending_cif_directory_path_after_images = None
            self._begin_async_cif_directory_index(str(Path(pending_directory)))
            return
        directory = self.cif_dir_edit.text().strip() if hasattr(self, "cif_dir_edit") else ""
        normalized_directory = str(Path(directory)) if directory else ""
        if normalized_directory and getattr(self, "_indexed_cif_directory", None) != normalized_directory:
            self._mark_thumbnail_grid_rebuild_pending()
            self._begin_async_cif_directory_index(normalized_directory)
            return
        self._schedule_thumbnail_grid_rebuild(force=True)

    def _on_cif_directory_index_finished(self, directory_state) -> None:
        if bool(getattr(self, "_image_list_rebuild_in_progress", False)):
            self._pending_cif_directory_state = directory_state
            return
        self._apply_cif_directory_state(directory_state)

    def _apply_pending_cif_directory_state_after_image_rebuild(self) -> bool:
        directory_state = getattr(self, "_pending_cif_directory_state", None)
        if directory_state is None:
            return False
        self._pending_cif_directory_state = None
        self._apply_cif_directory_state(directory_state)
        return True

    def _apply_cif_directory_state(self, directory_state) -> None:
        self.cif_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self._workspace.set_cif_index(directory_state.indexed_paths)
        self._indexed_cif_directory = directory_state.directory
        if directory_state.available:
            self._append_log(self._tr("cif_indexed_log", count=len(directory_state.indexed_paths)))
        else:
            self._append_log(self._tr("cif_directory_unavailable_log"))
        self._defer_vector_load_until_cif_index = False
        self._rebuild_vector_list()
        self._mark_thumbnail_grid_rebuild_pending()
        self._sync_after_cif_index_changed()

    def _on_cif_directory_index_failed(self, message: str) -> None:
        self._indexed_cif_directory = None
        self._append_log(
            self._tr(
                "cif_index_failed_log",
                "Не удалось индексировать векторы: {error}" if self._ui_language == "ru" else "Failed to index vectors: {error}",
                error=message,
            )
        )
        if not bool(getattr(self, "_image_list_rebuild_in_progress", False)):
            self._rebuild_thumbnail_grid()


