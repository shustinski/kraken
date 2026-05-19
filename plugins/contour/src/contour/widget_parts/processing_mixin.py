from __future__ import annotations

import cProfile
import hashlib
import io
import os
import pstats
from time import perf_counter, time_ns

from ._imports import *  # noqa: F403

PROFILE_CIF_OPEN = str(os.environ.get("CONTOUR_PROFILE_CIF_OPEN", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_PROFILE_CIF_OPEN_TOP_LINES = 40


class WidgetProcessingMixin:
    def _current_save_options(self) -> SaveOptions:
        return SaveOptions(
            save_cif=self.save_cif_checkbox.isChecked(),
            save_cv=self.save_cv_checkbox.isChecked(),
            save_preview=self.save_preview_checkbox.isChecked(),
        )

    def _paint_image_row_item(self, item: QListWidgetItem, image_path: str, *, show_text: bool = True) -> None:
        normalized = str(Path(image_path))
        extraction_enabled = self._is_extraction_mode_enabled()
        paint_image_row_item(
            item,
            normalized,
            image_has_changes=self._workspace.image_has_changes(normalized),
            has_vector_overlay=bool(self._workspace.resolve_cif_path(normalized)),
            vector_index_active=bool(self.cif_dir_edit.text().strip()) if hasattr(self, "cif_dir_edit") else False,
            extraction_enabled=extraction_enabled,
            viewed=normalized in self._viewed_image_paths,
            persisted_highlight=normalized in self._persisted_highlight_paths,
            show_text=show_text,
        )

    def _find_image_list_item(self, image_path: str) -> QListWidgetItem | None:
        target = str(Path(image_path))
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == target:
                return item
        return None

    def _update_frame_item_status(self, image_path: str | None) -> None:
        if not image_path:
            return
        item = self._find_image_list_item(image_path)
        if item is None:
            return
        self._paint_image_row_item(item, str(Path(image_path)))
        self._refresh_vector_items_for_stems({Path(str(image_path)).stem.lower()})
        self._update_thumbnail_item_status(image_path)

    def _refresh_image_list_item_states(self) -> None:
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is None:
                continue
            image_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if image_path:
                self._paint_image_row_item(item, image_path)
        self._refresh_vector_rows_for_workspace()
        self._update_thumbnail_grid_selection()

    def _update_thumbnail_item_status(self, image_path: str | None) -> None:
        if not image_path or not hasattr(self, "thumbnail_grid"):
            return
        normalized = str(Path(image_path))
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == normalized:
                self._paint_image_row_item(item, normalized, show_text=False)
                break

    def _update_vector_edit_status_label(self) -> None:
        if not hasattr(self, "vector_edit_status_label"):
            return
        if self._workspace.current_image_path is None:
            self.vector_edit_status_label.clear()
            return
        if not self._updating_views:
            self._workspace.update_current_polygons(self.get_polygons())
        dirty = self._workspace.current_image_has_changes()
        if self._ui_language == "ru":
            self.vector_edit_status_label.setText("Изменено" if dirty else "Сохранено")
        else:
            self.vector_edit_status_label.setText("Modified" if dirty else "Saved")

    def _persist_current_overlay_changes(self) -> bool:
        """Persist editor polygons for the current frame (dataset export and/or linked CIF)."""

        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            return True
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        if not self._workspace.current_image_has_changes():
            self._update_frame_item_status(current_image_path)
            self._update_vector_edit_status_label()
            return True

        want_dataset = bool(self.dataset_mode_checkbox.isChecked())
        can_cif = bool(current_state.loaded_cif_path and current_state.source_image is not None)

        if not want_dataset and not can_cif:
            self._append_log(
                self._tr(
                    "vector_save_no_target_log",
                    "Нет каталога набора данных или связанного CIF для сохранения правок текущего кадра."
                    if self._ui_language == "ru"
                    else "No dataset directory or linked CIF available to save edits for the current frame.",
                )
            )
            return False

        if want_dataset:
            saved_ds = self._export_dataset_frame_for_state(current_image_path, current_state, current_polygons)
            if not saved_ds:
                return False

        if can_cif:
            image_size = (int(current_state.source_image.shape[1]), int(current_state.source_image.shape[0]))
            try:
                save_polygons_vector(
                    current_state.loaded_cif_path,
                    current_image_path,
                    current_polygons,
                    image_size=image_size,
                )
            except Exception as exc:
                self._append_log(
                    self._tr(
                        "autosave_failed_log",
                        "Не удалось сохранить CIF {path}: {error}"
                        if self._ui_language == "ru"
                        else "Failed to save CIF {path}: {error}",
                        path=current_state.loaded_cif_path,
                        error=exc,
                    )
                )
                return False

        persisted_path = str(Path(current_image_path))
        self._persisted_highlight_paths.add(persisted_path)
        self._workspace.sync_polygon_reference_to_current(persisted_path)
        self._append_log(
            self._tr(
                "vectors_persisted_transition_log",
                "Изменения векторов сохранены для кадра {name}"
                if self._ui_language == "ru"
                else "Vector edits saved for frame {name}",
                name=Path(current_image_path).name,
            )
        )
        self._update_frame_item_status(current_image_path)
        self._update_vector_edit_status_label()
        return True

    def _discard_current_vector_changes(self) -> None:
        state = self._workspace.current_state
        path = self._workspace.current_image_path
        if state is None or path is None:
            return
        restored = [polygon.clone() for polygon in state.reference_polygons]
        self._updating_views = True
        try:
            state.polygons = restored
            self._workspace.update_current_polygons(restored)
            self.polygon_editor.set_polygons(restored)
        finally:
            self._updating_views = False
        self._persisted_highlight_paths.discard(str(Path(path)))
        self._update_frame_item_status(path)
        self._update_vector_edit_status_label()

    def _prompt_transition_vector_save_dialog(self) -> TransitionPromptChoice:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(
            self._tr(
                "unsaved_vectors_dialog_title",
                "Несохранённые изменения" if self._ui_language == "ru" else "Unsaved changes",
            )
        )
        msg.setText(
            self._tr(
                "unsaved_vectors_dialog_text",
                "Сохранить изменения?" if self._ui_language == "ru" else "Save changes?",
            )
        )
        save_button = msg.addButton(
            "Сохранить" if self._ui_language == "ru" else "Save",
            QMessageBox.ButtonRole.AcceptRole,
        )
        discard_button = msg.addButton(
            "Не сохранять" if self._ui_language == "ru" else "Don't save",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = msg.addButton(
            "Отмена" if self._ui_language == "ru" else "Cancel",
            QMessageBox.ButtonRole.RejectRole,
        )
        autosave_checkbox = QCheckBox(
            "Автосохранение при переходе к следующему кадру"
            if self._ui_language == "ru"
            else "Autosave on next frame"
        )
        autosave_checkbox.setChecked(False)
        msg.setCheckBox(autosave_checkbox)
        msg.setDefaultButton(save_button)
        result = msg.exec()
        clicked = msg.clickedButton()
        if clicked is None:
            if result == QMessageBox.StandardButton.Cancel:
                return TransitionPromptChoice.CANCEL
            if result == QMessageBox.StandardButton.Discard:
                return TransitionPromptChoice.DISCARD
            if result == QMessageBox.StandardButton.Save and autosave_checkbox.isChecked() and hasattr(
                self, "autosave_on_frame_transition_checkbox"
            ):
                self.autosave_on_frame_transition_checkbox.setChecked(True)
            return TransitionPromptChoice.SAVE
        if clicked is cancel_button:
            return TransitionPromptChoice.CANCEL
        if clicked is discard_button:
            return TransitionPromptChoice.DISCARD
        if clicked is save_button and autosave_checkbox.isChecked() and hasattr(
            self, "autosave_on_frame_transition_checkbox"
        ):
            self.autosave_on_frame_transition_checkbox.setChecked(True)
        return TransitionPromptChoice.SAVE

    def _warn_transition_blocked_after_failed_autosave(self) -> None:
        QMessageBox.warning(
            self,
            self._tr(
                "autosave_transition_blocked_title",
                "Не удалось сохранить" if self._ui_language == "ru" else "Save failed",
            ),
            self._tr(
                "autosave_transition_blocked_text",
                "Автосохранение не выполнено; переход отменён, данные не потеряны."
                if self._ui_language == "ru"
                else "Autosave failed; navigation cancelled — your edits were kept.",
            ),
        )

    def _warn_transition_blocked_after_failed_manual_save(self) -> None:
        QMessageBox.warning(
            self,
            self._tr(
                "manual_save_transition_blocked_title",
                "Не удалось сохранить" if self._ui_language == "ru" else "Save failed",
            ),
            self._tr(
                "manual_save_transition_blocked_text",
                "Сохранение не выполнено; переход отменён, данные не потеряны."
                if self._ui_language == "ru"
                else "Save failed; navigation cancelled — your edits were kept.",
            ),
        )

    def _try_leave_current_frame(self) -> bool:
        if self._is_extraction_mode_enabled():
            return True
        self._workspace.update_current_polygons(self.get_polygons())
        dirty = self._workspace.current_image_has_changes()
        if not dirty:
            self._update_vector_edit_status_label()
            return True

        autosave_on = bool(
            hasattr(self, "autosave_on_frame_transition_checkbox")
            and self.autosave_on_frame_transition_checkbox.isChecked()
        )
        if autosave_on:
            save_ok = self._persist_current_overlay_changes()
            allowed = navigation_allowed_after_autosave_attempt(dirty=True, save_ok=save_ok)
            if not allowed:
                self._warn_transition_blocked_after_failed_autosave()
            return allowed

        choice = self._prompt_transition_vector_save_dialog()
        if choice == TransitionPromptChoice.CANCEL:
            return False

        if choice == TransitionPromptChoice.DISCARD:
            self._discard_current_vector_changes()
            return True

        save_ok = self._persist_current_overlay_changes()
        allowed = navigation_allowed_after_prompt(dirty=True, choice=TransitionPromptChoice.SAVE, save_ok=save_ok)
        if not allowed:
            self._warn_transition_blocked_after_failed_manual_save()
        return allowed

    def confirm_ok_to_leave_current_vectors(self) -> bool:
        """Ask before closing or reloading images when the active frame has unsaved vector edits."""

        return self._try_leave_current_frame()

    def _sync_current_state_views(self) -> None:
        self._updating_views = True
        try:
            display_image = self._display_image_for_current_state()
            current_state = self._workspace.current_state
            polygons_synced = [polygon.clone() for polygon in current_state.polygons] if current_state else []
            self.polygon_editor.set_image(display_image)
            self.polygon_editor.set_polygons(polygons_synced)
            self.polygon_editor.set_debug_candidates(list(current_state.debug_candidates) if current_state else [])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
            if hasattr(self, "via_show_detected_checkbox"):
                self.polygon_editor.set_polygon_category_visible(
                    "via", self.via_show_detected_checkbox.isChecked()
                )
            if hasattr(self, "polygon_editor") and hasattr(self, "metal_show_rejected_checkbox"):
                layers = getattr(current_state, "metal_overlay_polygons", None) or {}
                self.polygon_editor.set_metal_overlays(
                    layers,
                    {
                        "rejected": self.metal_show_rejected_checkbox.isChecked(),
                        "suspicious": self.metal_show_suspicious_checkbox.isChecked(),
                        "border": self.metal_show_border_checkbox.isChecked(),
                        "wide_pairs_suspicious": self.metal_show_suspicious_checkbox.isChecked(),
                        "wide_pairs_rejected": self.metal_show_rejected_checkbox.isChecked(),
                    },
                )
            if hasattr(self, "metal_show_conductors_checkbox"):
                show_c = self.metal_show_conductors_checkbox.isChecked()
                self.polygon_editor.set_polygon_category_visible("conductor", show_c)
                self.polygon_editor.set_polygon_category_visible("metal_border", show_c)
                self.polygon_editor.set_polygon_category_visible("metal_wide_gradient", show_c)
            self._sync_neighbor_frames()
            self._sync_extra_layers()
            self._refresh_gradient_overlay()
        finally:
            self._updating_views = False
        self._update_vector_edit_status_label()

    def _display_image_for_current_state(self):
        current_state = self._workspace.current_state
        if self._show_source_while_middle_held and current_state is not None and current_state.source_image is not None:
            return current_state.source_image
        return self._workspace.current_display_image()

    def _via_debug_inspection_enabled(self) -> bool:
        return bool(hasattr(self, "debug_candidates_checkbox") and self.debug_candidates_checkbox.isChecked())

    def _neighbor_frame_image(self, image_path: str):
        state = getattr(self._workspace, "_state_cache", {}).get(image_path)
        if state is not None:
            return state.preprocessed_image if state.preprocessed_image is not None else state.source_image
        cached = self._neighbor_image_cache.get(image_path)
        if cached is not None:
            return cached
        try:
            image = load_image_color(image_path)
        except Exception as exc:
            self._append_log(
                self._tr(
                    "neighbor_frame_load_failed_log",
                    "Не удалось загрузить соседний кадр {path}: {error}"
                    if self._ui_language == "ru"
                    else "Failed to load neighbor frame {path}: {error}",
                    path=image_path,
                    error=exc,
                )
            )
            return None
        self._neighbor_image_cache[image_path] = image
        return image

    def _odd_neighbor_grid_size(self, value: int) -> int:
        size = max(3, min(7, int(value)))
        return size if size % 2 else size - 1

    def _neighbor_grid_size_for_zoom(self) -> int:
        max_grid = self._odd_neighbor_grid_size(self.neighbor_max_grid_spin.value())
        zoom = self.polygon_editor.zoom_factor() if hasattr(self, "polygon_editor") else 1.0
        requested = 7 if zoom < 0.25 else 5 if zoom < 0.45 else 3
        return min(max_grid, requested)

    def _sync_neighbor_frames(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        if not hasattr(self, "show_neighbor_frames_checkbox") or not self.show_neighbor_frames_checkbox.isChecked():
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        current_path = self._workspace.current_image_path
        image_paths = list(self._workspace.image_paths)
        if not current_path or current_path not in image_paths:
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        current_index = image_paths.index(current_path)
        columns = max(1, int(self.neighbor_columns_spin.value()))
        current_row = current_index // columns
        current_column = current_index % columns
        radius = self._neighbor_grid_size_for_zoom() // 2
        frames: list[tuple[int, int, object, str]] = []
        for row_offset in range(-radius, radius + 1):
            for column_offset in range(-radius, radius + 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                row = current_row + row_offset
                column = current_column + column_offset
                if row < 0 or column < 0 or column >= columns:
                    continue
                index = row * columns + column
                if index < 0 or index >= len(image_paths):
                    continue
                image_path = image_paths[index]
                image = self._neighbor_frame_image(image_path)
                if image is None:
                    continue
                frames.append((column_offset, row_offset, image, image_path))
        self.polygon_editor.set_neighbor_frames(
            frames,
            float(self.neighbor_opacity_spin.value()),
            int(self.neighbor_overlap_spin.value()),
            True,
        )

    def _on_neighbor_frame_activated(self, image_path: str) -> None:
        if image_path in self._workspace.image_paths:
            item = self._find_image_list_item(image_path)
            if item is not None:
                self.image_list.setCurrentItem(item)
            else:
                self.load_image(image_path)

    def _abort_in_flight_interactive_processing(self, *, preview: bool, prepared: bool) -> None:
        self._preview_update_timer.stop()
        if preview:
            if self._preview_run_cancel is not None:
                self._preview_run_cancel.set()
            self._preview_pending_request = None
            self._preview_pending_signature = None
        if prepared:
            if self._prepared_image_run_cancel is not None:
                self._prepared_image_run_cancel.set()
            self._prepared_image_pending_request = None
            self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()

    def _queue_prepared_image_update(self, image_path: str, source_image) -> None:
        request = PreparedImageRequest(
            image_path=image_path,
            source_image=source_image,
            pipeline_config=self.get_pipeline(),
        )
        signature = self._prepared_image_request_signature(request)
        if signature == self._prepared_image_running_signature or signature == self._prepared_image_pending_signature:
            self._refresh_busy_indicator()
            return
        if self._prepared_image_run_cancel is not None:
            self._prepared_image_run_cancel.set()
        self._prepared_image_pending_request = request
        self._prepared_image_pending_signature = signature
        self._refresh_busy_indicator()
        self._start_pending_prepared_image_update()

    def _start_pending_prepared_image_update(self) -> None:
        if self._prepared_image_pending_request is None:
            return
        if self._prepared_image_running_request_id is not None:
            if self._prepared_image_run_cancel is not None:
                self._prepared_image_run_cancel.set()
            return
        request = self._prepared_image_pending_request
        self._prepared_image_pending_request = None
        request_signature = self._prepared_image_pending_signature
        self._prepared_image_pending_signature = None
        self._prepared_image_request_serial += 1
        request_id = self._prepared_image_request_serial
        self._prepared_image_running_request_id = request_id
        self._prepared_image_running_signature = request_signature
        cancel = threading.Event()
        self._prepared_image_run_cancel = cancel
        worker = PreparedImageRunnable(request_id=request_id, request=request, cancel_event=cancel)
        worker.signals.result.connect(self._on_prepared_image_result)
        worker.signals.error.connect(self._on_prepared_image_error)
        worker.signals.finished.connect(self._on_prepared_image_finished)
        self._prepared_image_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _build_preview_request(self) -> PreviewProcessingRequest | None:
        if not self._workspace.current_image_path:
            return None
        source_image = None
        preprocessed_image = None
        current_state = self._workspace.current_state
        pipeline_config = self.get_pipeline()
        if current_state is not None and current_state.image_path == self._workspace.current_image_path:
            source_image = current_state.source_image
            if current_state.preprocessed_image is not None and current_state.pipeline_config == pipeline_config:
                preprocessed_image = current_state.preprocessed_image
        passthrough: tuple[PolygonData, ...] | None = None
        if hasattr(self, "recognition_mode_combo") and str(self.recognition_mode_combo.currentData() or "") == "disabled":
            passthrough = tuple(polygon.clone() for polygon in self.get_polygons())
        return PreviewProcessingRequest(
            image_path=self._workspace.current_image_path,
            pipeline_config=pipeline_config,
            contour_settings=self._current_contour_settings(),
            source_image=source_image,
            preprocessed_image=preprocessed_image,
            passthrough_polygons=passthrough,
        )

    def _preview_request_signature(self, request: PreviewProcessingRequest) -> tuple[str, str, str, int]:
        return build_preview_request_signature(request)

    def _prepared_image_request_signature(self, request: PreparedImageRequest) -> tuple[str, str]:
        return build_prepared_image_signature(request)

    def _queue_preview_processing(self, *, debounced: bool) -> None:
        request = self._build_preview_request()
        if request is None:
            self._append_log(self._tr("no_image_selected_log"))
            return
        if (
            normalize_via_search_mode(request.contour_settings.via_search_mode) == VIA_SEARCH_MODE_TEMPLATE
            and not list(getattr(request.contour_settings, "via_template_images", []) or [])
        ):
            self._append_log("Для режима поиска по шаблону добавьте хотя бы один шаблон")
        if hasattr(self, "recognition_mode_combo"):
            self._set_recognition_status("updating")
        signature = self._preview_request_signature(request)
        if signature == self._preview_running_signature or signature == self._preview_pending_signature:
            self._refresh_busy_indicator()
            return
        self._preview_update_timer.stop()
        if self._preview_run_cancel is not None:
            self._preview_run_cancel.set()
        self._preview_pending_request = request
        self._preview_pending_signature = signature
        self._refresh_busy_indicator()
        if debounced:
            self._preview_update_timer.start()
            return
        self._preview_update_timer.stop()
        self._start_pending_preview_processing()

    def _start_pending_preview_processing(self) -> None:
        if self._preview_pending_request is None:
            return
        if self._preview_running_request_id is not None:
            if self._preview_run_cancel is not None:
                self._preview_run_cancel.set()
            return
        request = self._preview_pending_request
        self._preview_pending_request = None
        request_signature = self._preview_pending_signature
        self._preview_pending_signature = None
        self._preview_request_serial += 1
        request_id = self._preview_request_serial
        self._preview_running_request_id = request_id
        self._preview_running_signature = request_signature
        cancel = threading.Event()
        self._preview_run_cancel = cancel
        self._reset_busy_progress(request)
        self._busy_progress_timer.start()
        worker = PreviewProcessingRunnable(request_id=request_id, request=request, cancel_event=cancel)
        worker.signals.result.connect(self._on_preview_processing_result)
        worker.signals.error.connect(self._on_preview_processing_error)
        worker.signals.finished.connect(self._on_preview_processing_finished)
        self._preview_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _append_log(self, message: str) -> None:
        self.logMessage.emit(message)

    def _refresh_busy_indicator(self) -> None:
        active = any(
            (
                self._preview_running_request_id is not None,
                self._preview_pending_request is not None,
                self._preview_update_timer.isActive(),
                self._prepared_image_running_request_id is not None,
                self._prepared_image_pending_request is not None,
                self._auto_tune_running_request_id is not None,
            )
        )
        if hasattr(self, "preview_busy_label"):
            suffix = f" — {self._busy_progress_value}%" if active and self._busy_progress_value > 0 else ""
            self.preview_busy_label.setText(f"{self._busy_indicator_text()}{suffix}")
            self.preview_busy_label.setVisible(active)
        if hasattr(self, "preview_busy_progress"):
            if active:
                self.preview_busy_progress.setValue(self._busy_progress_value)
            self.preview_busy_progress.setVisible(active)
        if active and self._preview_running_request_id is not None:
            if not self._busy_progress_timer.isActive():
                self._busy_progress_timer.start()
        else:
            self._busy_progress_timer.stop()
        if hasattr(self, "auto_tune_button"):
            self.auto_tune_button.setEnabled(self._auto_tune_running_request_id is None)

    def _on_prepared_image_result(
        self, request_id: int, image_path: str, preprocessed_image, pipeline_config: dict
    ) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        if pipeline_config != self.get_pipeline():
            return
        if self._workspace.store_preprocessed_image(image_path, preprocessed_image, pipeline_config):
            self._sync_current_state_views()
            self._try_extract_if_recognition_enabled()

    def _on_prepared_image_error(self, request_id: int, message: str) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_prepared_image_finished(self, request_id: int) -> None:
        if request_id == self._prepared_image_running_request_id:
            self._prepared_image_running_request_id = None
            self._prepared_image_running_signature = None
            self._prepared_image_run_cancel = None
        if self._prepared_image_pending_request is not None:
            self._start_pending_prepared_image_update()
        self._refresh_busy_indicator()

    def _on_auto_tune_result(self, request_id: int, result: AutoTuneResult) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._apply_auto_tune_result(result)
        roi_width = result.roi_bbox[2]
        roi_height = result.roi_bbox[3]
        self._append_log(
            self._tr(
                "auto_tune_finished_log",
                "Автоподбор завершён: score={score:.3f}, ROI={width}x{height}, проверок={evaluations}."
                if self._ui_language == "ru"
                else "Auto-fit completed: score={score:.3f}, ROI={width}x{height}, evaluations={evaluations}.",
                score=result.score,
                width=roi_width,
                height=roi_height,
                evaluations=result.evaluations,
            )
        )

    def _on_auto_tune_error(self, request_id: int, message: str) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._append_log(
            self._tr(
                "auto_tune_failed_log",
                "Ошибка автоподбора: {error}" if self._ui_language == "ru" else "Auto-fit failed: {error}",
                error=message,
            )
        )

    def _on_auto_tune_finished(self, request_id: int) -> None:
        if request_id == self._auto_tune_running_request_id:
            self._auto_tune_running_request_id = None
        self._refresh_busy_indicator()

    def _on_preview_processing_result(self, request_id: int, result) -> None:
        if request_id != self._preview_running_request_id:
            return
        if self._workspace.current_image_path != result.image_path:
            return

        if self._workspace.apply_processing_result(result):
            self._sync_current_state_views()
        self._update_frame_item_status(result.image_path)
        if hasattr(self, "recognition_mode_combo"):
            if str(self.recognition_mode_combo.currentData() or "") == "disabled":
                self._set_recognition_status("disabled")
            else:
                self._set_recognition_status("idle")
        self._set_progress_status("current_image_processed_status")
        self._append_log(
            self._tr(
                "current_image_processed_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )
        self.imageProcessed.emit(result.image_path, result.polygons)

    def _on_preview_processing_error(self, request_id: int, message: str) -> None:
        if request_id != self._preview_running_request_id:
            return
        if hasattr(self, "recognition_mode_combo"):
            self._set_recognition_status("error", message)
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_preview_processing_finished(self, request_id: int) -> None:
        if request_id == self._preview_running_request_id:
            self._preview_running_request_id = None
            self._preview_running_signature = None
            self._preview_run_cancel = None
            self._preview_running_request_for_progress = None
            self._busy_progress_stage = ""
            self._busy_progress_value = 0
        if self._preview_pending_request is not None and not self._preview_update_timer.isActive():
            self._start_pending_preview_processing()
        self._refresh_busy_indicator()

    def _show_batch_progress(self, total: int) -> None:
        if not self._batch_progress_enabled:
            self._hide_batch_progress()
            return
        self.batch_progress_bar.setRange(0, max(1, total))
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)

    def _hide_batch_progress(self) -> None:
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)

    def _on_polygons_edited(self) -> None:
        if self._updating_views:
            return
        if self._workspace.update_current_polygons(self.get_polygons()):
            current_path = self._workspace.current_image_path
            if current_path:
                self._persisted_highlight_paths.discard(str(Path(current_path)))
            self._update_frame_item_status(self._workspace.current_image_path)
            self._update_vector_edit_status_label()
            self.polygonsEdited.emit()

    def _antialias_selected_polygons(self) -> None:
        grade = int(self.antialias_grade_spin.value()) if hasattr(self, "antialias_grade_spin") else 1
        if not self.polygon_editor.antialias_selected_polygons(grade):
            self._append_log(
                self._tr(
                    "antialias_selected_none_log",
                    "Р’С‹Р±РµСЂРёС‚Рµ РїРѕР»РёРіРѕРЅС‹ РґР»СЏ СЃРіР»Р°Р¶РёРІР°РЅРёСЏ."
                    if self._ui_language == "ru"
                    else "Select polygons to antialias.",
                )
            )
            return
        self._append_log(
            self._tr(
                "antialias_selected_done_log",
                "РЎРіР»Р°Р¶РёРІР°РЅРёРµ РїСЂРёРјРµРЅРµРЅРѕ Рє РІС‹Р±СЂР°РЅРЅС‹Рј РїРѕР»РёРіРѕРЅР°Рј, grade={grade}."
                if self._ui_language == "ru"
                else "Antialiasing applied to selected polygons, grade={grade}.",
                grade=grade,
            )
        )

    def _antialias_opened_cif_files(self) -> None:
        grade = int(self.antialias_grade_spin.value()) if hasattr(self, "antialias_grade_spin") else 1
        current_path = self._workspace.current_image_path
        if current_path is not None:
            self._workspace.update_current_polygons(self.get_polygons())

        changed_count = 0
        saved_count = 0
        failed: list[str] = []
        for image_path, state in self._workspace.cached_states():
            if str(Path(image_path)) not in self._viewed_image_paths and image_path != current_path:
                continue
            if not state.loaded_cif_path or state.source_image is None:
                continue
            antialiased, changed = antialias_polygons(state.polygons, grade)
            if not changed:
                continue
            changed_count += 1
            image_size = (int(state.source_image.shape[1]), int(state.source_image.shape[0]))
            try:
                save_polygons_vector(state.loaded_cif_path, image_path, antialiased, image_size=image_size)
            except Exception as exc:
                failed.append(f"{Path(state.loaded_cif_path).name}: {exc}")
                continue
            state.polygons = [polygon.clone() for polygon in antialiased]
            state.reference_polygons = [polygon.clone() for polygon in antialiased]
            state.polygons_dirty = False
            self._persisted_highlight_paths.add(str(Path(image_path)))
            saved_count += 1
            if image_path == current_path:
                self._updating_views = True
                try:
                    self.polygon_editor.set_polygons(antialiased)
                finally:
                    self._updating_views = False
                self._update_vector_edit_status_label()
            self._update_frame_item_status(image_path)

        self._refresh_image_list_item_states()
        if failed:
            self._append_log(
                self._tr(
                    "antialias_opened_failed_log",
                    "РЎРіР»Р°Р¶РёРІР°РЅРёРµ CIF: СЃРѕС…СЂР°РЅРµРЅРѕ {saved}/{changed}, РѕС€РёР±РєРё: {errors}"
                    if self._ui_language == "ru"
                    else "CIF antialiasing: saved {saved}/{changed}, errors: {errors}",
                    saved=saved_count,
                    changed=changed_count,
                    errors="; ".join(failed[:4]),
                )
            )
            return
        self._append_log(
            self._tr(
                "antialias_opened_done_log",
                "РЎРіР»Р°Р¶РёРІР°РЅРёРµ РїСЂРёРјРµРЅРµРЅРѕ Рё СЃРѕС…СЂР°РЅРµРЅРѕ РґР»СЏ {count} РѕС‚РєСЂС‹С‚С‹С… CIF, grade={grade}."
                if self._ui_language == "ru"
                else "Antialiasing applied and saved for {count} opened CIF files, grade={grade}.",
                count=saved_count,
                grade=grade,
            )
        )

    def _on_batch_result(self, result) -> None:
        self.imageProcessed.emit(result.image_path, result.polygons)
        self._append_log(
            self._tr(
                "batch_result_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )

    def _on_batch_progress(self, current: int, total: int) -> None:
        if self._batch_progress_enabled:
            self.batch_progress_bar.setRange(0, max(1, total))
            self.batch_progress_bar.setValue(current)
        self._set_progress_status("batch_progress_status", current=current, total=total)
        self.batchProgress.emit(current, total)

    def _on_batch_finished(self) -> None:
        self._batch_progress_enabled = False
        self._hide_batch_progress()
        self._set_progress_status("batch_finished_status")
        self.batchFinished.emit()

    def _on_batch_error(self, image_path: str, message: str) -> None:
        self._append_log(self._tr("batch_error_log", image_name=Path(image_path).name, message=message))

    def refresh_image_list(self) -> None:
        directory = self.input_dir_edit.text().strip()
        if not directory:
            self._append_log(self._tr("input_directory_empty_log"))
            return
        self._begin_async_directory_scan(directory)

    def set_input_directory(self, path: str) -> None:
        directory = self._path_settings.validate_input_directory(path)
        if not directory.available:
            self._append_log(
                self._tr(
                    "input_directory_missing_log",
                    directory=directory.path,
                )
            )
            return
        self.input_dir_edit.setText(directory.path)
        self._save_persisted_paths()
        self._begin_async_directory_scan(directory.path)

    def set_cif_directory(self, path: str) -> None:
        directory_state = index_cif_directory(path)
        self.cif_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self._workspace.set_cif_index(directory_state.indexed_paths)
        if directory_state.available:
            self._append_log(self._tr("cif_indexed_log", count=len(directory_state.indexed_paths)))
        else:
            self._append_log(self._tr("cif_directory_unavailable_log"))
        self._sync_after_cif_index_changed()

    def set_output_directory(self, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()

    def set_dataset_directory(self, path: str) -> None:
        self.dataset_dir_edit.setText(path)
        self._save_persisted_paths()

    def _rebuild_image_list_items(self, normalized_paths: list[str]) -> None:
        self.image_list.clear()
        for path in normalized_paths:
            item = QListWidgetItem(Path(path).stem)
            item.setToolTip(f"Путь к файлу: {path}" if self._ui_language == "ru" else f"File path: {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._paint_image_row_item(item, path)
            self.image_list.addItem(item)
        self._rebuild_thumbnail_grid()

    def _select_loaded_image_path(self, image_path: str | None, *, fallback_to_first: bool = True) -> None:
        if image_path:
            item = self._find_image_list_item(image_path)
            if item is not None:
                self.image_list.setCurrentItem(item)
                return
        if fallback_to_first and self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
        elif self.image_list.count() <= 0:
            self._sync_current_state_views()

    def _apply_image_paths_to_workspace(
        self,
        paths: list[str],
        *,
        clear_extra_layers: bool,
        select_path: str | None,
        fallback_to_first: bool,
    ) -> None:
        frame_records, warnings = build_base_frame_records(paths)
        for message in warnings:
            self._append_log(message)
        ordered_paths = [record.path for record in frame_records]
        self._base_frame_number_by_path = build_base_frame_number_map(frame_records)
        self._base_frame_numbers = set(self._base_frame_number_by_path.values())
        normalized_paths = self._workspace.replace_image_selection(ordered_paths, is_supported_image=is_image_path)
        if not normalized_paths:
            self._save_persisted_current_image_path(None)
        self._neighbor_image_cache.clear()
        self._prune_tagged_sets_for_images(normalized_paths)
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        if clear_extra_layers:
            self._clear_extra_layers()
        self._update_extra_layers_enabled_state()
        self._rebuild_image_list_items(normalized_paths)
        self._select_loaded_image_path(select_path, fallback_to_first=fallback_to_first)
        self._rebuild_vector_list()
        self._refresh_vector_rows_for_workspace()
        self._sync_frame_navigation_controls()
        self._log_matching_gaps_after_refresh(self._matching_report())

    def load_images(self, paths: list[str], *, preferred_current_image_path: str | None = None) -> None:
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._directory_scanner.invalidate_pending_results()
        normalized_paths = [str(Path(path)) for path in paths if is_image_path(path)]
        preferred = str(Path(preferred_current_image_path)) if preferred_current_image_path else None
        if preferred not in normalized_paths:
            preferred = None
        self._apply_image_paths_to_workspace(
            normalized_paths,
            clear_extra_layers=True,
            select_path=preferred,
            fallback_to_first=True,
        )
        return

    def append_images(self, paths: list[str], *, select_first_new: bool = True) -> None:
        existing_paths = [str(Path(path)) for path in self._workspace.image_paths]
        existing_set = set(existing_paths)
        additions: list[str] = []
        seen = set(existing_set)
        for path in paths:
            normalized = str(Path(path))
            if normalized in seen or not is_visible_image_path(normalized):
                continue
            seen.add(normalized)
            additions.append(normalized)
        if not additions:
            return
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._directory_scanner.invalidate_pending_results()
        select_path = additions[0] if select_first_new else self._workspace.current_image_path
        self._apply_image_paths_to_workspace(
            [*existing_paths, *additions],
            clear_extra_layers=False,
            select_path=select_path,
            fallback_to_first=not bool(self._workspace.current_image_path),
        )

    def reset_project(self) -> None:
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._directory_scanner.invalidate_pending_results()
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self._workspace.clear_project()
        self._base_frame_number_by_path = {}
        self._base_frame_numbers = set()
        self._neighbor_image_cache.clear()
        self._persisted_highlight_paths.clear()
        self._viewed_image_paths.clear()
        self._cif_load_failure_stems.clear()
        self.input_dir_edit.setText("")
        self.cif_dir_edit.setText("")
        self._save_persisted_paths()
        self._save_persisted_current_image_path(None)
        self.image_list.clear()
        self._rebuild_thumbnail_grid()
        self._clear_extra_layers()
        self._update_extra_layers_enabled_state()
        self._rebuild_vector_list()
        self._refresh_image_list_item_states()
        self._sync_frame_navigation_controls()
        self._sync_current_state_views()

    def _find_matching_cif_path(self, image_path: str) -> str | None:
        return self._workspace.resolve_cif_path(image_path)

    def _load_cif_overlay_polygons(self, image_path: str) -> list[PolygonData]:
        stem_key = Path(image_path).stem.lower()
        cif_path = self._find_matching_cif_path(image_path)
        if not cif_path:
            self._cif_load_failure_stems.discard(stem_key)
            return []
        try:
            referenced_image, image_size, polygons = load_polygons_vector(cif_path)
        except Exception as exc:
            self._cif_load_failure_stems.add(stem_key)
            self._append_log(self._tr("cif_load_failed_log", file_name=Path(cif_path).name, error=exc))
            return []
        self._cif_load_failure_stems.discard(stem_key)
        if referenced_image and Path(referenced_image).stem.lower() != Path(image_path).stem.lower():
            self._append_log(
                self._tr(
                    "cif_reference_name_diff_log",
                    file_name=Path(cif_path).name,
                    referenced_image=referenced_image,
                )
            )
        if image_size is not None:
            self._append_log(
                self._tr(
                    "cif_overlay_loaded_with_size_log",
                    file_name=Path(cif_path).name,
                    width=image_size[0],
                    height=image_size[1],
                    count=len(polygons),
                )
            )
        else:
            self._append_log(self._tr("cif_overlay_loaded_log", file_name=Path(cif_path).name, count=len(polygons)))
        return polygons

    def load_image(self, path: str) -> None:
        normalized_load_path = str(Path(path))
        active_load_path = getattr(self, "_loading_image_path", None)
        if active_load_path is not None:
            if active_load_path == normalized_load_path:
                return
            QTimer.singleShot(0, lambda queued_path=normalized_load_path: self.load_image(queued_path))
            return
        self._loading_image_path = normalized_load_path
        cif_path_for_profile = self._find_matching_cif_path(path)
        profile_enabled = PROFILE_CIF_OPEN and bool(cif_path_for_profile)
        profiler = cProfile.Profile() if profile_enabled else None
        profile_timings: dict[str, float] = {}
        profile_total_start = perf_counter()
        try:
            if profiler is not None:
                profiler.enable()
            phase_start = perf_counter()
            self._abort_in_flight_interactive_processing(preview=True, prepared=True)
            if profile_enabled:
                profile_timings["abort_in_flight"] = (perf_counter() - phase_start) * 1000.0
                phase_start = perf_counter()

            def load_source_image_timed(image_path: str):
                inner_start = perf_counter()
                try:
                    return load_image_color(image_path)
                finally:
                    if profile_enabled:
                        profile_timings["source_image_load"] = (perf_counter() - inner_start) * 1000.0

            def load_cif_overlay_timed(image_path: str) -> list[PolygonData]:
                inner_start = perf_counter()
                try:
                    return self._load_cif_overlay_polygons(image_path)
                finally:
                    if profile_enabled:
                        profile_timings["cif_overlay_load"] = (perf_counter() - inner_start) * 1000.0

            image_result = self._workspace.load_image(
                normalized_load_path,
                load_source_image=load_source_image_timed,
                load_cif_overlay=load_cif_overlay_timed,
            )
            self._save_persisted_current_image_path(image_result.image_path)
            if profile_enabled:
                profile_timings["workspace_load"] = (perf_counter() - phase_start) * 1000.0
                phase_start = perf_counter()
            if not self._is_extraction_mode_enabled():
                self._viewed_image_paths.add(str(Path(image_result.image_path)))
                self._handle_gamification_ui_event(RewardEventType.IMAGE_VIEWED)
            if image_result.state is not None and not image_result.cache_hit and not image_result.reused_current_state:
                image_result.state.loaded_cif_path = self._find_matching_cif_path(image_result.image_path)
                image_result.state.reference_polygons = [polygon.clone() for polygon in image_result.state.polygons]
                image_result.state.polygons_dirty = False
            if image_result.reused_current_state:
                self._update_frame_item_status(image_result.image_path)
                self._update_thumbnail_grid_selection()
                self._sync_frame_navigation_controls()
                self.polygon_editor.center_main_image()
                if profile_enabled:
                    profile_timings["sync_reused"] = (perf_counter() - phase_start) * 1000.0
                    profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
                    if profiler is not None:
                        profiler.disable()
                    self._emit_cif_open_profile(
                        profile_timings,
                        image_path=image_result.image_path,
                        cif_path=cif_path_for_profile,
                        polygon_count=0 if image_result.state is None else len(image_result.state.polygons),
                        profiler=profiler,
                    )
                return
            self._sync_current_state_views()
            self.polygon_editor.center_main_image()
            self._update_frame_item_status(image_result.image_path)
            self._update_thumbnail_grid_selection()
            self._sync_frame_navigation_controls()
            if profile_enabled:
                profile_timings["sync_views"] = (perf_counter() - phase_start) * 1000.0
                phase_start = perf_counter()
            if (
                image_result.prepared_image_required
                and image_result.state is not None
                and image_result.state.source_image is not None
            ):
                self._queue_prepared_image_update(image_result.image_path, image_result.state.source_image)
            if profile_enabled:
                profile_timings["queue_prepared"] = (perf_counter() - phase_start) * 1000.0
                phase_start = perf_counter()
            if image_result.cache_hit:
                self._append_log(self._tr("loaded_cached_state_log", image_path=image_result.image_path))
            else:
                self._append_log(self._tr("loaded_image_log", image_path=image_result.image_path))
            if profile_enabled:
                profile_timings["log"] = (perf_counter() - phase_start) * 1000.0
                phase_start = perf_counter()
            self._try_extract_if_recognition_enabled()
            if profile_enabled:
                profile_timings["maybe_extract"] = (perf_counter() - phase_start) * 1000.0
                profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
                if profiler is not None:
                    profiler.disable()
                self._emit_cif_open_profile(
                    profile_timings,
                    image_path=image_result.image_path,
                    cif_path=cif_path_for_profile,
                    polygon_count=0 if image_result.state is None else len(image_result.state.polygons),
                    profiler=profiler,
                )
        finally:
            if profiler is not None:
                profiler.disable()
            if getattr(self, "_loading_image_path", None) == normalized_load_path:
                self._loading_image_path = None

    def _emit_cif_open_profile(
        self,
        timings_ms: dict[str, float],
        *,
        image_path: str,
        cif_path: str,
        polygon_count: int,
        profiler: cProfile.Profile | None,
    ) -> None:
        if not PROFILE_CIF_OPEN:
            return
        total_ms = timings_ms.get("total_wall", sum(timings_ms.values()))
        detail = " ".join(
            f"{name}={elapsed:.3f}ms" for name, elapsed in timings_ms.items() if name != "total_wall"
        )
        message = (
            f"[contour cif open profiling] total={total_ms:.3f}ms polygons={polygon_count} "
            f"image={Path(image_path).name} cif={Path(cif_path).name} {detail}"
        )
        print(message)
        self._append_log(message)
        if profiler is None:
            return
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(_PROFILE_CIF_OPEN_TOP_LINES)
        report = stream.getvalue()
        print(f"[contour cif open profiling stats] top={_PROFILE_CIF_OPEN_TOP_LINES}")
        print(report)
        self._append_log(f"[contour cif open profiling stats] top={_PROFILE_CIF_OPEN_TOP_LINES}\n{report}")

    def _is_extraction_mode_enabled(self) -> bool:
        if not hasattr(self, "recognition_mode_combo"):
            return False
        return str(self.recognition_mode_combo.currentData() or "") != "disabled"

    def get_polygons(self) -> list[PolygonData]:
        return self.polygon_editor.get_polygons()

    def set_pipeline(self, config: dict) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(config)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def get_pipeline(self) -> dict:
        return self._pipeline.to_dict()

    def process_current_image(self, *_args, debounced: bool = False) -> None:
        self._queue_preview_processing(debounced=debounced)

    def _export_dataset_frame_for_state(
        self,
        image_path: str,
        state: ImageProcessingState,
        polygons: list[PolygonData],
        dataset_directory: str | None = None,
    ) -> dict[str, str]:
        target_directory = dataset_directory or self.dataset_dir_edit.text().strip()
        result = export_frame_to_dataset(
            dataset_directory=target_directory,
            image_path=image_path,
            state=state,
            polygons=polygons,
        )
        if result.message_key is not None:
            self._append_log(self._tr(result.message_key, **(result.message_kwargs or {})))
        return result.saved_files

    def export_current_frame_to_dataset(self, dataset_directory: str | None = None) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        self._update_frame_item_status(current_image_path)
        return self._export_dataset_frame_for_state(
            current_image_path,
            current_state,
            current_polygons,
            dataset_directory=dataset_directory,
        )

    def save_current_result(
        self,
        output_directory: str | None = None,
        save_options: SaveOptions | None = None,
    ) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        target_directory = output_directory or self.output_dir_edit.text().strip()
        if not target_directory:
            self._append_log(self._tr("output_directory_not_set_log"))
            return {}
        self._workspace.update_current_polygons(self.get_polygons())
        had_vector_edits = self._workspace.current_image_has_changes()
        saved_files = save_result_bundle(
            output_directory=target_directory,
            image_path=current_image_path,
            polygons=self.get_polygons(),
            source_image=current_state.source_image,
            display_settings=self._display_settings,
            save_options=save_options or self._current_save_options(),
            metadata={
                "contour_settings": self._current_contour_settings().to_dict(),
                "pipeline": self.get_pipeline(),
            },
        )
        if saved_files:
            self._append_log(self._tr("saved_result_log", saved_files=saved_files))
            self._handle_gamification_after_save(
                image_path=current_image_path,
                state=current_state,
                polygons=self.get_polygons(),
                saved_files=saved_files,
                had_vector_edits=had_vector_edits,
            )
            saved_key = str(Path(current_image_path))
            if had_vector_edits:
                self._persisted_highlight_paths.add(saved_key)
            self._workspace.sync_polygon_reference_to_current(saved_key)
            self._update_frame_item_status(current_image_path)
            self._update_vector_edit_status_label()
        return saved_files

    def start_batch_processing(
        self,
        image_paths: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        if self._batch_controller.is_running:
            self._append_log(self._tr("batch_already_running_log"))
            return
        paths = image_paths or list(self._workspace.image_paths)
        if not paths:
            self._append_log(self._tr("batch_no_images_log"))
            return
        output_directory = self.output_dir_edit.text().strip() or None
        save_options = self._current_save_options()
        started = self._batch_controller.start(
            BatchStartRequest(
                image_paths=list(paths),
                pipeline_config=self.get_pipeline(),
                contour_settings=self._current_contour_settings(),
                display_settings=self._display_settings,
                save_options=save_options,
                output_directory=output_directory,
                max_workers=max_workers or self.max_workers_spin.value(),
                )
            )
        if not started:
            return
        self._batch_progress_enabled = self._batch_controller.progress_enabled
        self._show_batch_progress(len(paths))
        self._set_progress_status("batch_started_status")

    def stop_batch_processing(self) -> None:
        self._batch_controller.stop()

    def _handle_gamification_after_save(
        self,
        *,
        image_path: str,
        state: ImageProcessingState,
        polygons: list[PolygonData],
        saved_files: dict[str, str],
        had_vector_edits: bool,
    ) -> None:
        if not hasattr(self, "_gamification_service"):
            return
        try:
            reference_polygons = list(getattr(state, "reference_polygons", []))
            event = CorrectionEvent(
                correction_id=self._build_gamification_correction_id(
                    image_path=image_path,
                    polygons=polygons,
                    saved_files=saved_files,
                ),
                image_id=str(Path(image_path)),
                has_real_mask_changes=bool(had_vector_edits),
                accepted_without_changes=not bool(had_vector_edits),
                correction_type=CorrectionType.UNKNOWN if had_vector_edits else None,
                edit_count=None,
                added_objects=max(0, len(polygons) - len(reference_polygons)),
                removed_objects=max(0, len(reference_polygons) - len(polygons)),
            )
            results = self._gamification_service.handle_correction_event(event)
            if hasattr(self, "gamification_panel"):
                if results:
                    self.gamification_panel.set_last_results(results)
                elif event.accepted_without_changes:
                    self.gamification_panel.react_to_event(RewardEventType.IMAGE_ACCEPTED_WITHOUT_CHANGES)
            else:
                for result in results:
                    if result.success and result.message:
                        self._append_log(result.message)
        except Exception as exc:
            self._append_log(f"Gamification error: {exc}")

    def _handle_gamification_ui_event(self, event_type: RewardEventType) -> None:
        try:
            if hasattr(self, "gamification_panel"):
                self.gamification_panel.react_to_event(event_type)
        except Exception as exc:
            self._append_log(f"Gamification UI error: {exc}")

    @staticmethod
    def _build_gamification_correction_id(
        *,
        image_path: str,
        polygons: list[PolygonData],
        saved_files: dict[str, str],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(str(Path(image_path)).encode("utf-8", errors="replace"))
        for polygon in sorted(polygons, key=lambda item: int(item.id)):
            digest.update(
                (
                    f"|{polygon.id}:{polygon.is_hole}:{polygon.parent_id}:"
                    f"{polygon.category}:{polygon.shape_hint}:"
                    f"{polygon.bbox}:{len(polygon.points)}"
                ).encode("utf-8", errors="replace")
            )
            for x_coord, y_coord in polygon.points:
                digest.update(f":{float(x_coord):.6f},{float(y_coord):.6f}".encode("ascii"))
        for key, value in sorted(saved_files.items()):
            digest.update(f"|{key}:{value}".encode("utf-8", errors="replace"))
        return f"{Path(image_path).stem}:{time_ns()}:{digest.hexdigest()[:16]}"
