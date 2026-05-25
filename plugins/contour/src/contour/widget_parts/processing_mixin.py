from __future__ import annotations

import cProfile
import hashlib
import shutil
from time import perf_counter, time_ns

from typing import Any

from PyQt6.QtGui import QImage, QImageReader

from ..infrastructure.frame_switch_profiler import (
    MAX_IDLE_POLLS,
    FrameSwitchProfile,
    frame_switch_profiling_enabled,
    profile_callable,
)
from ._imports import *  # noqa: F403


class WidgetProcessingMixin:
    def _current_save_options(self: Any) -> SaveOptions:
        return SaveOptions(
            save_cif=self.save_cif_checkbox.isChecked(),
            save_cv=self.save_cv_checkbox.isChecked(),
            save_preview=self.save_preview_checkbox.isChecked(),
        )

    def _paint_image_row_item(self: Any, item: QListWidgetItem, image_path: str, *, show_text: bool = True) -> None:
        normalized = str(Path(image_path))
        extraction_enabled = self._is_extraction_mode_enabled()
        paint_image_row_item(
            item,
            normalized,
            image_has_changes=self._workspace.image_has_changes(normalized),
            has_vector_overlay=bool(self._workspace.resolve_cif_path(normalized)),
            vector_index_active=self._vector_index_active(),
            extraction_enabled=extraction_enabled,
            viewed=normalized in self._viewed_image_paths,
            persisted_highlight=normalized in self._persisted_highlight_paths,
            theme=getattr(self, "_ui_theme", "dark"),
            show_text=show_text,
        )

    def _uses_large_frame_list(self: Any) -> bool:
        image_paths = getattr(self._workspace, "_image_paths", None)
        if image_paths is None:
            image_paths = self._workspace.image_paths
        return len(image_paths) > LARGE_FRAME_COUNT_THRESHOLD

    def _image_path_index(self: Any, image_path: str | Path) -> int | None:
        return self._image_path_to_index.get(str(Path(image_path)))

    def _image_list_model_item_data(self: Any, image_path: str, role: int):
        from ..application.frame_asset_sync import (
            background_hex_image_paint_status_for_theme,
            classify_image_side_paint_status,
            foreground_hex_image_paint_status_for_theme,
        )

        normalized = (
            image_path
            if image_path in self._image_path_to_index
            else str(Path(image_path))
        )
        cif_paths = getattr(self._workspace, "_cif_paths_by_stem", {})
        stem_lower = getattr(self, "_image_path_stem_lower", {}).get(normalized)
        if stem_lower is None:
            stem_lower = Path(normalized).stem.lower()
        has_vector = stem_lower in cif_paths
        extraction_enabled = self._is_extraction_mode_enabled()
        current_path = self._workspace.current_image_path
        if extraction_enabled or current_path == normalized:
            polygons_dirty = False if extraction_enabled else self._workspace.image_has_changes(normalized)
        elif self._uses_large_frame_list():
            polygons_dirty = False
        else:
            polygons_dirty = self._workspace.image_has_changes(normalized)
        painted = classify_image_side_paint_status(
            has_matching_cif=has_vector,
            vector_index_active=self._vector_index_active(),
            never_opened=True if extraction_enabled else normalized not in self._viewed_image_paths,
            polygons_dirty=polygons_dirty,
            persist_highlight=False if extraction_enabled else normalized in self._persisted_highlight_paths,
        )
        theme = getattr(self, "_ui_theme", "dark")
        if role == FRAME_STATUS_ROLE:
            return painted.value
        if role == int(Qt.ItemDataRole.BackgroundRole):
            hex_background = background_hex_image_paint_status_for_theme(painted, theme=theme)
            return QColor(hex_background) if hex_background else None
        if role == int(Qt.ItemDataRole.ForegroundRole):
            return QColor(
                foreground_hex_image_paint_status_for_theme(
                    painted,
                    has_matching_cif=has_vector,
                    theme=theme,
                )
            )
        return None

    def _set_image_list_paths(self: Any, normalized_paths: list[str]) -> None:
        paths = [str(Path(path)) for path in normalized_paths]
        if hasattr(self.image_list, "clear_manual_items"):
            self.image_list.clear_manual_items()
        self._image_path_to_index = {path: index for index, path in enumerate(paths)}
        self._image_path_stem_lower = {path: Path(path).stem.lower() for path in paths}
        self._image_list_model.set_paths(paths)

    def _thumbnail_disk_cache_marker_path(self: Any) -> Path:
        return Path(getattr(self, "_thumbnail_disk_cache_dir", Path())) / "cache.key"

    def _thumbnail_disk_cache_key_for_base_paths(self: Any, paths: list[str]) -> str:
        if not paths:
            return ""
        return str(Path(paths[0]).parent)

    def _clear_thumbnail_disk_cache(self: Any) -> None:
        if hasattr(self, "_cancel_thumbnail_loading"):
            self._cancel_thumbnail_loading()
        cache_dir = Path(getattr(self, "_thumbnail_disk_cache_dir", Path()))
        if not cache_dir:
            return
        try:
            shutil.rmtree(cache_dir)
        except FileNotFoundError:
            pass
        except Exception:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._thumbnail_loaded_generation.clear()
        self._thumbnail_loaded_sizes.clear()
        self._thumbnail_queued_paths.clear()
        self._thumbnail_queued_sizes.clear()
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        getattr(self, "_thumbnail_pending_apply", {}).clear()

    def _reset_thumbnail_disk_cache_for_base_paths(self: Any, paths: list[str], *, force: bool = False) -> None:
        cache_dir = Path(getattr(self, "_thumbnail_disk_cache_dir", Path()))
        if not cache_dir:
            return
        key = self._thumbnail_disk_cache_key_for_base_paths(paths)
        marker_path = self._thumbnail_disk_cache_marker_path()
        previous_key = ""
        try:
            previous_key = marker_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            previous_key = ""
        except Exception:
            previous_key = ""
        if force or (previous_key and previous_key != key):
            self._clear_thumbnail_disk_cache()
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            marker_path.write_text(key, encoding="utf-8")
        except Exception:
            pass
        self._thumbnail_disk_cache_key = key

    def _image_list_source_index(self: Any, proxy_index: QModelIndex) -> QModelIndex:
        if not proxy_index.isValid():
            return QModelIndex()
        return self._image_list_proxy.mapToSource(proxy_index)

    def _image_list_path_from_proxy_index(self: Any, proxy_index: QModelIndex) -> str | None:
        source_index = self._image_list_source_index(proxy_index)
        if not source_index.isValid():
            return None
        value = self._image_list_model.data(source_index, Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _image_list_selected_paths(self: Any) -> list[str]:
        paths: list[str] = []
        selection = self.image_list.selectionModel()
        if selection is None:
            return paths
        for proxy_index in selection.selectedIndexes():
            path = self._image_list_path_from_proxy_index(proxy_index)
            if path:
                paths.append(path)
        return paths

    def _set_image_list_current_path(self: Any, image_path: str | None, *, fallback_to_first: bool = True) -> None:
        if image_path:
            row = self._image_path_index(image_path)
            if row is not None:
                proxy_index = self._image_list_proxy.mapFromSource(self._image_list_model.index(row))
                if proxy_index.isValid():
                    self.image_list.setCurrentIndex(proxy_index)
                return
        if fallback_to_first and self._image_list_proxy.rowCount() > 0:
            self.image_list.setCurrentIndex(self._image_list_proxy.index(0, 0))
        elif self._image_list_proxy.rowCount() <= 0:
            self._sync_current_state_views()

    def _image_path_in_image_list(self: Any, image_path: str) -> bool:
        return self._image_path_index(image_path) is not None

    def _update_frame_item_status(self: Any, image_path: str | None) -> None:
        if not image_path:
            return
        self._image_list_model.invalidate_path(image_path)
        self._refresh_vector_items_for_stems({Path(str(image_path)).stem.lower()})
        self._update_thumbnail_item_status(image_path)
        self._update_asset_items_for_image_path(image_path)

    def _refresh_image_list_item_states(self: Any) -> None:
        if self._uses_large_frame_list():
            current = self._workspace.current_image_path
            if current:
                self._image_list_model.invalidate_path(current)
        else:
            self._image_list_model.invalidate_all_rows()
        self._refresh_vector_rows_for_workspace()
        if len(self._workspace.image_paths) <= ASSET_FILTER_LISTS_MAX_FRAMES:
            self._rebuild_asset_filter_lists()
            self._apply_asset_view_filter()
        self._update_thumbnail_grid_selection()

    def _update_thumbnail_item_status(self: Any, image_path: str | None) -> None:
        if not image_path or not self._frame_matrix_enabled() or not hasattr(self, "thumbnail_grid"):
            return
        normalized = str(Path(image_path))
        row = self._thumbnail_row_for_path(normalized) if hasattr(self, "_thumbnail_row_for_path") else None
        if row is None:
            return
        item = self.thumbnail_grid.item(row)
        if item is None:
            return
        self._paint_image_row_item(item, normalized, show_text=False)

    def _update_vector_edit_status_label(self: Any, *, sync_editor: bool = True) -> None:
        if not hasattr(self, "vector_edit_status_label"):
            return
        if self._workspace.current_image_path is None:
            self.vector_edit_status_label.clear()
            return
        if sync_editor and not self._updating_views:
            self._sync_editor_polygons_to_current_workspace()
        dirty = self._workspace.current_image_has_changes()
        if self._ui_language == "ru":
            self.vector_edit_status_label.setText("Изменено" if dirty else "Сохранено")
        else:
            self.vector_edit_status_label.setText("Modified" if dirty else "Saved")

    def _persist_current_overlay_changes(self: Any) -> bool:
        """Persist editor polygons for the current frame (dataset export and/or linked CIF)."""

        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            return True
        if self._editor_polygons_are_current_frame():
            current_polygons = self.get_polygons()
            self._workspace.update_current_polygons(current_polygons)
        else:
            current_polygons = [polygon.clone() for polygon in current_state.polygons]
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

    def _discard_current_vector_changes(self: Any) -> None:
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

    def _prompt_transition_vector_save_dialog(self: Any) -> TransitionPromptChoice:
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

    def _warn_transition_blocked_after_failed_autosave(self: Any) -> None:
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

    def _warn_transition_blocked_after_failed_manual_save(self: Any) -> None:
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

    def _try_leave_current_frame(self: Any) -> bool:
        if self._is_extraction_mode_enabled():
            return True
        self._sync_editor_polygons_to_current_workspace()
        dirty = self._workspace.current_image_has_changes()
        if not dirty:
            self._update_vector_edit_status_label(sync_editor=False)
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

    def confirm_ok_to_leave_current_vectors(self: Any) -> bool:
        """Ask before closing or reloading images when the active frame has unsaved vector edits."""

        return self._try_leave_current_frame()

    def _editor_display_cache_key(self: Any, image_path: str, state) -> tuple[str, str]:
        if state is not None and state.preprocessed_image is not None:
            return (str(Path(image_path)), "preprocessed")
        return (str(Path(image_path)), "source")

    def _polygons_editor_signature(self: Any, image_path: str, polygons: list) -> tuple[object, ...]:
        geometry = []
        for polygon in polygons:
            points = getattr(polygon, "points", ())
            first_point = points[0] if points else None
            last_point = points[-1] if points else None
            geometry.append(
                (
                    getattr(polygon, "id", None),
                    len(points),
                    getattr(polygon, "bbox", None),
                    first_point,
                    last_point,
                    getattr(polygon, "category", None),
                    getattr(polygon, "shape_hint", None),
                    bool(getattr(polygon, "is_hole", False)),
                    getattr(polygon, "parent_id", None),
                )
            )
        return (str(Path(image_path)), tuple(geometry))

    def _sync_polygons_to_editor(self: Any, image_path: str, polygons: list) -> None:
        signature = self._polygons_editor_signature(image_path, polygons)
        if getattr(self, "_editor_polygons_signature", None) == signature:
            if len(self.polygon_editor.get_polygons()) == len(polygons):
                return
        self.polygon_editor.set_polygons(polygons, emit_signal=False)
        self._editor_polygons_signature = signature

    def _editor_polygons_are_current_frame(self: Any) -> bool:
        current_path = self._workspace.current_image_path
        if not current_path:
            return False
        if getattr(self, "_pending_editor_frame_apply", None) is not None:
            return False
        signature = getattr(self, "_editor_polygons_signature", None)
        if not isinstance(signature, tuple) or not signature:
            return False
        return str(Path(signature[0])) == str(Path(current_path))

    def _sync_editor_polygons_to_current_workspace(self: Any) -> bool:
        if not self._editor_polygons_are_current_frame():
            return False
        return self._workspace.update_current_polygons(self.get_polygons())

    def _neighbor_frames_enabled(self: Any) -> bool:
        return bool(
            hasattr(self, "show_neighbor_frames_checkbox")
            and self.show_neighbor_frames_checkbox.isChecked()
        )

    def _neighbor_vectors_enabled(self: Any) -> bool:
        return bool(
            self._neighbor_frames_enabled()
            and hasattr(self, "show_neighbor_vectors_checkbox")
            and self.show_neighbor_vectors_checkbox.isChecked()
        )

    def _request_neighbor_frame_sync(self: Any, *, delay_ms: int = 0) -> None:
        if not self._neighbor_frames_enabled():
            timer = getattr(self, "_neighbor_sync_timer", None)
            if timer is not None:
                timer.stop()
            self._sync_neighbor_frames()
            return
        timer = getattr(self, "_neighbor_sync_timer", None)
        if timer is None:
            QTimer.singleShot(max(0, int(delay_ms)), self._sync_neighbor_frames)
            return
        timer.stop()
        timer.start(max(0, int(delay_ms)))

    def _schedule_neighbor_frames_after_main_image_ready(self: Any) -> None:
        self._request_neighbor_frame_sync(delay_ms=0)

    def _clear_neighbor_frame_display_for_frame_change(self: Any) -> None:
        self._neighbor_sync_image_path = None
        self._neighbor_frame_specs = []
        self._neighbor_queued_paths.clear()
        timer = getattr(self, "_neighbor_apply_timer", None)
        if timer is not None:
            timer.stop()
        if not hasattr(self, "polygon_editor"):
            return
        scene = getattr(self.polygon_editor, "_editor_scene", None)
        if scene is None:
            return
        scene._pending_neighbor_frames = None
        scene.clear_neighbor_frames()

    def _neighbor_sync_is_current(self: Any) -> bool:
        sync_path = getattr(self, "_neighbor_sync_image_path", None)
        current_path = self._workspace.current_image_path
        if not sync_path or not current_path:
            return False
        return str(Path(sync_path)) == str(Path(current_path))

    def _queue_editor_display_pixmap(self: Any, image_path: str, display_image: object) -> None:
        if display_image is None:
            self.polygon_editor.set_image_pixmap(QPixmap())
            return
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.mark_pending("editor_display")
        self._editor_display_request_serial = int(getattr(self, "_editor_display_request_serial", 0)) + 1
        request_id = self._editor_display_request_serial
        target_path = str(Path(image_path))
        runnable = EditorDisplayRunnable(request_id, target_path, display_image)

        def _on_display_ready(req_id: int, path: str, qimage: object) -> None:
            if req_id != self._editor_display_request_serial:
                return
            if str(Path(path)) != str(Path(self._workspace.current_image_path or "")):
                return
            pixmap = QPixmap()
            if isinstance(qimage, QImage):
                try:
                    pixmap = QPixmap.fromImage(qimage)
                except Exception:
                    pixmap = QPixmap()
            if pixmap.isNull():
                self.polygon_editor.set_image_pixmap(QPixmap())
                self._pending_editor_frame_apply = None
                return
            current_state = self._workspace.current_state
            if current_state is not None:
                cache_key = self._editor_display_cache_key(path, current_state)
                cache = getattr(self, "_editor_pixmap_cache", {})
                cache[cache_key] = pixmap
                while len(cache) > 48:
                    cache.pop(next(iter(cache)))
                self._editor_pixmap_cache = cache
            self.polygon_editor.set_image_pixmap(pixmap)
            session = self._frame_switch_profile_for_path(path)
            if session is not None:
                session.complete_pending("editor_display", suffix="_ready")
            self._flush_pending_editor_frame_apply(str(Path(path)))

        runnable.signals.result.connect(_on_display_ready)
        self._editor_display_thread_pool.start(runnable)

    def _apply_display_image_to_editor(self: Any, image_path: str, display_image: object, *, state) -> bool:
        if display_image is None:
            self.polygon_editor.set_image_pixmap(QPixmap())
            return True
        cache_key = self._editor_display_cache_key(image_path, state)
        cached = getattr(self, "_editor_pixmap_cache", {}).get(cache_key)
        if cached is not None and not cached.isNull():
            self.polygon_editor.set_image_pixmap(cached)
            return True
        self._queue_editor_display_pixmap(image_path, display_image)
        return False

    def _apply_editor_vectors_for_frame(
        self: Any,
        image_path: str,
        state,
        polygons: list,
        *,
        defer_heavy_overlays: bool,
    ) -> None:
        self._sync_polygons_to_editor(image_path, polygons)
        if defer_heavy_overlays:
            return
        self.polygon_editor.set_debug_candidates(list(state.debug_candidates))
        self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
        if hasattr(self, "via_show_detected_checkbox"):
            self.polygon_editor.set_polygon_category_visible(
                "via", self.via_show_detected_checkbox.isChecked()
            )
        if hasattr(self, "polygon_editor") and hasattr(self, "metal_show_rejected_checkbox"):
            layers = getattr(state, "metal_overlay_polygons", None) or {}
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

    def _flush_pending_editor_frame_apply(self: Any, image_path: str) -> None:
        pending = getattr(self, "_pending_editor_frame_apply", None)
        if pending is None:
            session = self._frame_switch_profile_for_path(image_path)
            if session is not None:
                session.complete_pending("editor_vectors", suffix="_none_pending")
            self._schedule_neighbor_frames_after_main_image_ready()
            return
        pending_path, polygons, defer_heavy_overlays = pending
        self._pending_editor_frame_apply = None
        if str(Path(pending_path)) != str(Path(image_path)):
            return
        if str(Path(pending_path)) != str(Path(self._workspace.current_image_path or "")):
            return
        current_state = self._workspace.current_state
        if current_state is None:
            return
        step_start = perf_counter()
        self._apply_editor_vectors_for_frame(
            pending_path,
            current_state,
            polygons,
            defer_heavy_overlays=defer_heavy_overlays,
        )
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.note_timing("editor_vectors_apply", (perf_counter() - step_start) * 1000.0)
            session.complete_pending("editor_vectors", suffix="_flushed")
        self._schedule_neighbor_frames_after_main_image_ready()

    def _apply_frame_to_editor(self: Any, *, defer_neighbors: bool = True, defer_heavy_overlays: bool = False) -> None:
        current_state = self._workspace.current_state
        image_path = self._workspace.current_image_path
        self._clear_neighbor_frame_display_for_frame_change()
        if current_state is None or not image_path:
            self._pending_editor_frame_apply = None
            self.polygon_editor.set_image_pixmap(QPixmap())
            self.polygon_editor.set_polygons([], emit_signal=False)
            self._editor_polygons_signature = None
            return
        display_image = self._display_image_for_current_state()
        polygons = list(current_state.polygons)
        image_ready = self._apply_display_image_to_editor(image_path, display_image, state=current_state)
        if image_ready:
            self._pending_editor_frame_apply = None
            self._apply_editor_vectors_for_frame(
                image_path,
                current_state,
                polygons,
                defer_heavy_overlays=defer_heavy_overlays,
            )
        else:
            self._pending_editor_frame_apply = (str(Path(image_path)), polygons, defer_heavy_overlays)
            session = self._frame_switch_profile_for_path(image_path)
            if session is not None:
                session.mark_pending("editor_vectors")
        neighbor_delay_ms = 200 if self._uses_large_frame_list() else 0
        if defer_neighbors:
            self._request_neighbor_frame_sync(delay_ms=neighbor_delay_ms)
        else:
            self._request_neighbor_frame_sync(delay_ms=0)
        if not defer_heavy_overlays:
            self._sync_extra_layers()
            self._refresh_gradient_overlay()
        self._update_vector_edit_status_label()

    def _sync_current_state_views(self: Any, *, defer_neighbors: bool = True) -> None:
        self._updating_views = True
        try:
            self._apply_frame_to_editor(defer_neighbors=defer_neighbors, defer_heavy_overlays=False)
        finally:
            self._updating_views = False

    def _display_image_for_current_state(self: Any):
        current_state = self._workspace.current_state
        if self._show_source_while_middle_held and current_state is not None and current_state.source_image is not None:
            return current_state.source_image
        return self._workspace.current_display_image()

    def _via_debug_inspection_enabled(self: Any) -> bool:
        return bool(hasattr(self, "debug_candidates_checkbox") and self.debug_candidates_checkbox.isChecked())

    def _neighbor_preview_max_dimension(self: Any) -> int:
        state = self._workspace.current_state
        if state is not None and state.source_image is not None:
            height, width = state.source_image.shape[:2]
            return max(128, min(512, int(min(width, height))))
        return 256

    def _neighbor_frame_image(self: Any, image_path: str):
        normalized = str(Path(image_path))
        state = getattr(self._workspace, "_state_cache", {}).get(normalized)
        if state is not None:
            return state.preprocessed_image if state.preprocessed_image is not None else state.source_image
        cached = self._neighbor_image_cache.get(normalized)
        if cached is not None:
            return cached
        return None

    def _queue_neighbor_frame_load(self: Any, image_path: str) -> None:
        normalized = str(Path(image_path))
        if normalized in getattr(self, "_neighbor_queued_paths", set()):
            return
        self._neighbor_queued_paths.add(normalized)
        try:
            max_dim = self._neighbor_preview_max_dimension()
            runnable = ThumbnailLoadRunnable(0, normalized, max_dim, max_dim)
            runnable.signals.result.connect(
                lambda generation, image_path, width, height, qimage: self._on_neighbor_frame_loaded(
                    generation,
                    image_path,
                    width,
                    height,
                    qimage,
                ),
                Qt.ConnectionType.QueuedConnection,
            )
            runnable.signals.finished.connect(
                self._on_neighbor_frame_load_finished,
                Qt.ConnectionType.QueuedConnection,
            )
            self._neighbor_thread_pool.start(runnable)
        except RuntimeError:
            self._neighbor_queued_paths.discard(normalized)

    def _on_neighbor_frame_loaded(
        self: Any,
        _generation: int,
        image_path: str,
        _width: int,
        _height: int,
        qimage: object,
    ) -> None:
        normalized = str(Path(image_path))
        self._neighbor_queued_paths.discard(normalized)
        if not self._neighbor_sync_is_current():
            return
        if not any(str(Path(path)) == normalized for *_offsets, path in getattr(self, "_neighbor_frame_specs", [])):
            return
        if qimage is None:
            return
        self._neighbor_image_cache[normalized] = qimage
        self._schedule_neighbor_frame_apply(delay_ms=0)

    def _on_neighbor_frame_load_finished(self: Any, _generation: int, image_path: str) -> None:
        self._neighbor_queued_paths.discard(str(Path(image_path)))
        if self._neighbor_sync_is_current() and not getattr(self, "_neighbor_queued_paths", set()):
            self._schedule_neighbor_frame_apply(delay_ms=0)

    def _schedule_neighbor_frame_apply(self: Any, *, delay_ms: int = 0) -> None:
        if not self._neighbor_sync_is_current():
            return
        timer = getattr(self, "_neighbor_apply_timer", None)
        if timer is None:
            QTimer.singleShot(max(0, int(delay_ms)), self._apply_cached_neighbor_frames)
            return
        timer.stop()
        timer.start(max(0, int(delay_ms)))

    def _try_load_neighbor_preview_image(self: Any, image_path: str):
        return self._neighbor_frame_image(str(image_path))

    def _neighbor_source_size(self: Any, image_path: str, image: object) -> tuple[int, int]:
        normalized = str(Path(image_path))
        cached = getattr(self, "_neighbor_image_dimensions", {}).get(normalized)
        if cached is not None:
            return max(1, int(cached[0])), max(1, int(cached[1]))
        state = getattr(self._workspace, "_state_cache", {}).get(normalized)
        source = getattr(state, "source_image", None) if state is not None else None
        if source is not None:
            height, width = source.shape[:2]
            size = (max(1, int(width)), max(1, int(height)))
            self._neighbor_image_dimensions[normalized] = size
            return size
        reader_size = QImageReader(normalized).size()
        if reader_size.isValid():
            size = (max(1, int(reader_size.width())), max(1, int(reader_size.height())))
            self._neighbor_image_dimensions[normalized] = size
            return size
        if isinstance(image, QImage) and not image.isNull():
            return max(1, image.width()), max(1, image.height())
        width = int(getattr(image, "shape", (1, 1))[1]) if hasattr(image, "shape") else 1
        height = int(getattr(image, "shape", (1, 1))[0]) if hasattr(image, "shape") else 1
        return max(1, width), max(1, height)

    def _neighbor_frame_vectors(self: Any, image_path: str) -> tuple[list[PolygonData], tuple[int, int] | None]:
        if not self._neighbor_vectors_enabled():
            return [], None
        normalized = str(Path(image_path))
        cache = getattr(self, "_neighbor_vector_cache", {})
        cached = cache.get(normalized)
        if cached is not None:
            return cached
        cif_path = self._find_matching_cif_path(normalized)
        if not cif_path:
            cache[normalized] = ([], None)
            self._neighbor_vector_cache = cache
            return [], None
        try:
            _referenced_image, image_size, polygons = load_polygons_vector(cif_path)
        except Exception:
            polygons = []
            image_size = None
        source_size = None
        if image_size is not None:
            source_size = (max(1, int(image_size[0])), max(1, int(image_size[1])))
        cached = ([polygon.clone() for polygon in polygons], source_size)
        cache[normalized] = cached
        if len(cache) > 256:
            cache.pop(next(iter(cache)))
        self._neighbor_vector_cache = cache
        return cached

    def _apply_cached_neighbor_frames(self: Any) -> None:
        if not self._neighbor_frames_enabled():
            self._neighbor_sync_image_path = None
            self._neighbor_frame_specs = []
            self._neighbor_queued_paths.clear()
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        if not self._neighbor_sync_is_current():
            return
        frames: list[tuple[int, int, object, str, list[PolygonData], tuple[int, int]]] = []
        include_vectors = self._neighbor_vectors_enabled()
        for column_offset, row_offset, image_path in getattr(self, "_neighbor_frame_specs", []):
            image = self._try_load_neighbor_preview_image(image_path)
            if image is None:
                continue
            source_size = self._neighbor_source_size(image_path, image)
            polygons: list[PolygonData] = []
            if include_vectors:
                polygons, vector_source_size = self._neighbor_frame_vectors(image_path)
                if vector_source_size is not None:
                    source_size = vector_source_size
            frames.append((column_offset, row_offset, image, image_path, polygons, source_size))
        if not frames and getattr(self, "_neighbor_queued_paths", set()):
            return
        self.polygon_editor.set_neighbor_frames(
            frames,
            float(self.neighbor_opacity_spin.value()),
            int(self.neighbor_overlap_spin.value()),
            True,
        )
        if getattr(self, "_neighbor_queued_paths", set()):
            self._schedule_neighbor_frame_apply(delay_ms=100)
        else:
            current_path = self._workspace.current_image_path
            session = (
                self._frame_switch_profile_for_path(str(current_path))
                if current_path
                else None
            )
            if session is not None:
                session.complete_pending("neighbor_sync", suffix="_applied")
            QTimer.singleShot(0, self._center_editor_on_current_main_image)

    def _center_editor_on_current_main_image(self: Any) -> None:
        if not hasattr(self, "polygon_editor") or not self._workspace.current_image_path:
            return
        self.polygon_editor.center_main_image()

    def _odd_neighbor_grid_size(self: Any, value: int) -> int:
        size = max(3, min(7, int(value)))
        return size if size % 2 else size - 1

    def _neighbor_grid_size(self: Any) -> int:
        return self._odd_neighbor_grid_size(self.neighbor_max_grid_spin.value())

    def _sync_neighbor_frames(self: Any) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        current_path = self._workspace.current_image_path
        session = (
            self._frame_switch_profile_for_path(str(current_path))
            if current_path
            else None
        )
        if session is not None:
            session.mark_pending("neighbor_sync")
        if not self._neighbor_frames_enabled():
            self._neighbor_sync_image_path = None
            self._neighbor_frame_specs = []
            self._neighbor_queued_paths.clear()
            timer = getattr(self, "_neighbor_apply_timer", None)
            if timer is not None:
                timer.stop()
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            if session is not None:
                session.complete_pending("neighbor_sync", suffix="_disabled")
            QTimer.singleShot(0, self._center_editor_on_current_main_image)
            return
        image_paths = [str(Path(path)) for path in self._workspace.image_paths]
        normalized_current_path = str(Path(current_path)) if current_path else ""
        if not normalized_current_path or normalized_current_path not in image_paths:
            self._neighbor_sync_image_path = None
            self._neighbor_frame_specs = []
            self._neighbor_queued_paths.clear()
            timer = getattr(self, "_neighbor_apply_timer", None)
            if timer is not None:
                timer.stop()
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            if session is not None:
                session.complete_pending("neighbor_sync", suffix="_no_current")
            QTimer.singleShot(0, self._center_editor_on_current_main_image)
            return
        current_index = image_paths.index(normalized_current_path)
        columns = max(1, int(self.neighbor_columns_spin.value()))
        current_row = current_index // columns
        current_column = current_index % columns
        radius = self._neighbor_grid_size() // 2
        previous_sync_path = getattr(self, "_neighbor_sync_image_path", None)
        self._neighbor_sync_image_path = normalized_current_path
        if str(previous_sync_path or "") != normalized_current_path:
            self._neighbor_queued_paths.clear()
            timer = getattr(self, "_neighbor_apply_timer", None)
            if timer is not None:
                timer.stop()
        scene = self.polygon_editor._editor_scene
        scene._pending_neighbor_frames = None
        scene.clear_neighbor_frames()
        specs: list[tuple[int, int, str]] = []
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
                specs.append((column_offset, row_offset, image_path))
        self._neighbor_frame_specs = specs
        spec_paths = {str(Path(path)) for _column_offset, _row_offset, path in specs}
        self._neighbor_queued_paths.intersection_update(spec_paths)
        self._apply_cached_neighbor_frames()
        for _column_offset, _row_offset, image_path in specs:
            if self._neighbor_frame_image(image_path) is None:
                self._queue_neighbor_frame_load(image_path)
        if getattr(self, "_neighbor_queued_paths", set()):
            self._schedule_neighbor_frame_apply(delay_ms=50)
        elif session is not None:
            session.complete_pending("neighbor_sync", suffix="_cached_only")
        if self._uses_large_frame_list() and len(self._neighbor_image_cache) > 48:
            self._neighbor_image_cache.clear()
            self._neighbor_image_dimensions.clear()

    def _on_neighbor_frame_activated(self: Any, image_path: str) -> None:
        if image_path in self._workspace.image_paths:
            if self._image_path_in_image_list(image_path):
                self._set_image_list_current_path(image_path, fallback_to_first=False)
            else:
                self.load_image(image_path)

    def _abort_in_flight_interactive_processing(self: Any, *, preview: bool, prepared: bool) -> None:
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

    def _queue_prepared_image_update(self: Any, image_path: str, source_image) -> None:
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

    def _start_pending_prepared_image_update(self: Any) -> None:
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

    def _build_preview_request(self: Any) -> PreviewProcessingRequest | None:
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

    def _preview_request_signature(self: Any, request: PreviewProcessingRequest) -> tuple[str, str, str, int]:
        return build_preview_request_signature(request)

    def _prepared_image_request_signature(self: Any, request: PreparedImageRequest) -> tuple[str, str]:
        return build_prepared_image_signature(request)

    def _queue_preview_processing(self: Any, *, debounced: bool) -> None:
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

    def _start_pending_preview_processing(self: Any) -> None:
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

    def _append_log(self: Any, message: str) -> None:
        self.logMessage.emit(message)

    def _refresh_busy_indicator(self: Any) -> None:
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
        self: Any, request_id: int, image_path: str, preprocessed_image, pipeline_config: dict
    ) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        if pipeline_config != self.get_pipeline():
            return
        if self._workspace.store_preprocessed_image(image_path, preprocessed_image, pipeline_config):
            self._sync_current_state_views()
            self._try_extract_if_recognition_enabled()

    def _on_prepared_image_error(self: Any, request_id: int, message: str) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_prepared_image_finished(self: Any, request_id: int) -> None:
        if request_id == self._prepared_image_running_request_id:
            self._prepared_image_running_request_id = None
            self._prepared_image_running_signature = None
            self._prepared_image_run_cancel = None
        if self._prepared_image_pending_request is not None:
            self._start_pending_prepared_image_update()
        self._refresh_busy_indicator()

    def _on_auto_tune_result(self: Any, request_id: int, result: AutoTuneResult) -> None:
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

    def _on_auto_tune_error(self: Any, request_id: int, message: str) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._append_log(
            self._tr(
                "auto_tune_failed_log",
                "Ошибка автоподбора: {error}" if self._ui_language == "ru" else "Auto-fit failed: {error}",
                error=message,
            )
        )

    def _on_auto_tune_finished(self: Any, request_id: int) -> None:
        if request_id == self._auto_tune_running_request_id:
            self._auto_tune_running_request_id = None
        self._refresh_busy_indicator()

    def _on_preview_processing_result(self: Any, request_id: int, result) -> None:
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

    def _on_preview_processing_error(self: Any, request_id: int, message: str) -> None:
        if request_id != self._preview_running_request_id:
            return
        if hasattr(self, "recognition_mode_combo"):
            self._set_recognition_status("error", message)
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_preview_processing_finished(self: Any, request_id: int) -> None:
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

    def _show_batch_progress(self: Any, total: int) -> None:
        if not self._batch_progress_enabled:
            self._hide_batch_progress()
            return
        self.batch_progress_bar.setRange(0, max(1, total))
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)

    def _hide_batch_progress(self: Any) -> None:
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)

    def _on_polygons_edited(self: Any) -> None:
        if self._updating_views:
            return
        if self._sync_editor_polygons_to_current_workspace():
            current_path = self._workspace.current_image_path
            if current_path:
                self._persisted_highlight_paths.discard(str(Path(current_path)))
            self._update_frame_item_status(self._workspace.current_image_path)
            self._update_vector_edit_status_label()
            self.polygonsEdited.emit()

    def _antialias_selected_polygons(self: Any) -> None:
        grade = int(self.antialias_grade_spin.value()) if hasattr(self, "antialias_grade_spin") else 1
        if not self.polygon_editor.antialias_selected_polygons(grade):
            self._append_log(
                self._tr(
                    "antialias_selected_none_log",
                    "Выберите полигоны для сглаживания."
                    if self._ui_language == "ru"
                    else "Select polygons to antialias.",
                )
            )
            return
        self._append_log(
            self._tr(
                "antialias_selected_done_log",
                "Сглаживание применено к выбранным полигонам, grade={grade}."
                if self._ui_language == "ru"
                else "Antialiasing applied to selected polygons, grade={grade}.",
                grade=grade,
            )
        )

    def _antialias_opened_cif_files(self: Any) -> None:
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
                    "Сглаживание CIF: сохранено {saved}/{changed}, ошибки: {errors}"
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
                "Сглаживание применено и сохранено для {count} открытых CIF, grade={grade}."
                if self._ui_language == "ru"
                else "Antialiasing applied and saved for {count} opened CIF files, grade={grade}.",
                count=saved_count,
                grade=grade,
            )
        )

    def _on_batch_result(self: Any, result) -> None:
        polygons = list(getattr(result, "polygons", []) or [])
        # Multiprocessing batch workers return metadata only to keep IPC and the
        # GUI event queue bounded during multi-thousand-image runs.
        self.imageProcessed.emit(result.image_path, polygons)
        self._append_log(
            self._tr(
                "batch_result_log",
                image_name=Path(result.image_path).name,
                count=int(getattr(result, "polygon_count", len(polygons))),
            )
        )

    def _on_batch_progress(self: Any, current: int, total: int) -> None:
        if self._batch_progress_enabled:
            self.batch_progress_bar.setRange(0, max(1, total))
            self.batch_progress_bar.setValue(current)
        self._set_progress_status("batch_progress_status", current=current, total=total)
        self.batchProgress.emit(current, total)

    def _on_batch_finished(self: Any) -> None:
        self._batch_progress_enabled = False
        self._hide_batch_progress()
        self._set_progress_status("batch_finished_status")
        self.batchFinished.emit()

    def _on_batch_error(self: Any, image_path: str, message: str) -> None:
        self._append_log(self._tr("batch_error_log", image_name=Path(image_path).name, message=message))

    def refresh_image_list(self: Any) -> None:
        directory = self.input_dir_edit.text().strip()
        if not directory:
            self._append_log(self._tr("input_directory_empty_log"))
            return
        self._begin_async_directory_scan(directory)

    def set_input_directory(self: Any, path: str) -> None:
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

    def set_cif_directory(self: Any, path: str) -> None:
        directory = str(Path(path))
        self.cif_dir_edit.setText(directory)
        self._save_persisted_paths()
        self._indexed_cif_directory = None
        if bool(getattr(self, "_image_list_rebuild_in_progress", False)):
            self._pending_cif_directory_path_after_images = directory
            return
        self._begin_async_cif_directory_index(directory)

    def set_output_directory(self: Any, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()

    def set_dataset_directory(self: Any, path: str) -> None:
        self.dataset_dir_edit.setText(path)
        self._save_persisted_paths()

    def _rebuild_image_list_items(self: Any, normalized_paths: list[str]) -> None:
        self._rebuild_image_list_items_responsive(normalized_paths)
        return
        self.image_list.clear()
        for path in normalized_paths:
            item = QListWidgetItem(Path(path).stem)
            item.setToolTip(f"Путь к файлу: {path}" if self._ui_language == "ru" else f"File path: {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._paint_image_row_item(item, path)
            self.image_list.addItem(item)
        self._rebuild_thumbnail_grid()

    def _rebuild_image_list_items_responsive(self: Any, normalized_paths: list[str]) -> None:
        self._image_list_build_generation += 1
        self._image_list_rebuild_in_progress = True
        paths = [str(Path(path)) for path in normalized_paths]
        if hasattr(self, "image_vector_list") and len(paths) <= ASSET_FILTER_LISTS_MAX_FRAMES:
            for list_widget in (self.image_vector_list, self.image_only_list, self.vector_only_list):
                list_widget.clear()
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(False)
            self.files_scan_progress_bar.setRange(0, 100)
            self.files_scan_progress_bar.setValue(0)
        self._set_image_list_paths(paths)
        if len(paths) <= ASSET_FILTER_LISTS_MAX_FRAMES:
            self._rebuild_asset_filter_lists()
            self._apply_asset_view_filter()
        self._finish_pending_image_list_rebuild()

    def _finish_pending_image_list_rebuild(self: Any) -> None:
        self._image_list_rebuild_in_progress = False
        directory = self.cif_dir_edit.text().strip() if hasattr(self, "cif_dir_edit") else ""
        normalized_directory = str(Path(directory)) if directory else ""
        self._defer_vector_load_until_cif_index = bool(
            normalized_directory and getattr(self, "_indexed_cif_directory", None) != normalized_directory
        )
        pending = getattr(self, "_pending_image_list_post_build", None)
        if not pending:
            if self._defer_vector_load_until_cif_index:
                self._mark_thumbnail_grid_rebuild_pending()
            if not self._apply_pending_cif_directory_state_after_image_rebuild():
                self._start_vector_index_or_rebuild_frame_matrix_after_images()
            return
        self._pending_image_list_post_build = None
        select_path = pending.get("select_path")
        self._select_loaded_image_path(
            str(select_path) if select_path else None,
            fallback_to_first=bool(pending.get("fallback_to_first", True)),
        )
        self._refresh_vector_rows_for_workspace()
        if not self._uses_large_frame_list():
            self._log_matching_gaps_after_refresh(self._matching_report())
        if self._defer_vector_load_until_cif_index:
            self._mark_thumbnail_grid_rebuild_pending()
        if not self._apply_pending_cif_directory_state_after_image_rebuild():
            self._start_vector_index_or_rebuild_frame_matrix_after_images()

    def _select_loaded_image_path(self: Any, image_path: str | None, *, fallback_to_first: bool = True) -> None:
        self._set_image_list_current_path(image_path, fallback_to_first=fallback_to_first)

    def _apply_image_paths_to_workspace(
        self: Any,
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
        self._reset_thumbnail_disk_cache_for_base_paths(normalized_paths)
        if not normalized_paths:
            self._save_persisted_current_image_path(None)
        self._neighbor_image_cache.clear()
        self._neighbor_image_dimensions.clear()
        self._neighbor_vector_cache.clear()
        self._prune_tagged_sets_for_images(normalized_paths)
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        if clear_extra_layers:
            self._clear_extra_layers()
        self._update_extra_layers_enabled_state()
        self._pending_image_list_post_build = {
            "select_path": select_path,
            "fallback_to_first": fallback_to_first,
        }
        self._rebuild_image_list_items(normalized_paths)
        return

    def load_images(self: Any, paths: list[str], *, preferred_current_image_path: str | None = None) -> None:
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._directory_scanner.invalidate_pending_results()
        self._stop_work_simulation()
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

    def append_images(self: Any, paths: list[str], *, select_first_new: bool = True) -> None:
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
        self._stop_work_simulation()
        select_path = additions[0] if select_first_new else self._workspace.current_image_path
        self._apply_image_paths_to_workspace(
            [*existing_paths, *additions],
            clear_extra_layers=False,
            select_path=select_path,
            fallback_to_first=not bool(self._workspace.current_image_path),
        )

    def reset_project(self: Any) -> None:
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._directory_scanner.invalidate_pending_results()
        self._vector_indexer.invalidate_pending_results()
        self._stop_work_simulation()
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self._workspace.clear_project()
        self._reset_thumbnail_disk_cache_for_base_paths([], force=True)
        self._base_frame_number_by_path = {}
        self._base_frame_numbers = set()
        self._neighbor_image_cache.clear()
        self._neighbor_image_dimensions.clear()
        self._neighbor_vector_cache.clear()
        self._persisted_highlight_paths.clear()
        self._viewed_image_paths.clear()
        self._cif_load_failure_stems.clear()
        self._editor_pixmap_cache.clear()
        self._editor_polygons_signature = None
        self._thumbnail_path_to_row.clear()
        if hasattr(self, "_cancel_thumbnail_loading"):
            self._cancel_thumbnail_loading()
        getattr(self, "_thumbnail_icon_cache", {}).clear()
        self.input_dir_edit.setText("")
        self.cif_dir_edit.setText("")
        self._save_persisted_paths()
        self._save_persisted_current_image_path(None)
        self._set_image_list_paths([])
        self._rebuild_thumbnail_grid()
        self._clear_extra_layers()
        self._update_extra_layers_enabled_state()
        self._rebuild_vector_list()
        self._refresh_image_list_item_states()
        self._sync_current_state_views()

    def _find_matching_cif_path(self: Any, image_path: str) -> str | None:
        return self._workspace.resolve_cif_path(image_path)

    def _should_defer_vector_load(self: Any) -> bool:
        return bool(getattr(self, "_defer_vector_load_until_cif_index", False))

    def _flush_pending_thumbnail_grid_rebuild(self: Any) -> None:
        if not self._frame_matrix_enabled():
            self._pending_thumbnail_rebuild_after_vectors = False
            self._thumbnail_flush_retry_count = 0
            return
        if not bool(getattr(self, "_pending_thumbnail_rebuild_after_vectors", False)):
            return
        if getattr(self, "_frame_load_running_path", None) is not None or getattr(self, "_loading_image_path", None):
            attempts = int(getattr(self, "_thumbnail_flush_retry_count", 0)) + 1
            self._thumbnail_flush_retry_count = attempts
            if attempts > 600:
                self._pending_thumbnail_rebuild_after_vectors = False
                self._thumbnail_flush_retry_count = 0
                QTimer.singleShot(0, self._rebuild_thumbnail_grid)
                return
            QTimer.singleShot(100, self._flush_pending_thumbnail_grid_rebuild)
            return
        self._pending_thumbnail_rebuild_after_vectors = False
        self._thumbnail_flush_retry_count = 0
        QTimer.singleShot(0, self._rebuild_thumbnail_grid)

    def _mark_thumbnail_grid_rebuild_pending(self: Any) -> None:
        if not self._frame_matrix_enabled():
            self._pending_thumbnail_rebuild_after_vectors = False
            self._thumbnail_flush_retry_count = 0
            return
        self._pending_thumbnail_rebuild_after_vectors = True
        self._thumbnail_flush_retry_count = 0

    def _defer_frame_chrome_updates(self: Any, image_path: str) -> None:
        normalized = str(Path(image_path))
        session = self._frame_switch_profile_for_path(normalized)
        if session is not None:
            session.mark_pending("deferred_chrome")
        if not hasattr(self, "_frame_chrome_update_timer"):
            self._frame_chrome_update_timer = QTimer(self)
            self._frame_chrome_update_timer.setSingleShot(True)
            self._frame_chrome_update_timer.timeout.connect(self._apply_deferred_frame_chrome_updates)
        self._pending_frame_chrome_path = normalized
        self._frame_chrome_update_timer.stop()
        self._frame_chrome_update_timer.start(0 if not self._uses_large_frame_list() else 32)

    def _apply_deferred_frame_chrome_updates(self: Any) -> None:
        image_path = str(getattr(self, "_pending_frame_chrome_path", "") or "")
        if not image_path or str(Path(self._workspace.current_image_path or "")) != image_path:
            return
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.complete_pending("deferred_chrome", suffix="_apply")
        self._image_list_model.invalidate_path(image_path)
        self._update_thumbnail_grid_selection()
        if not self._thumbnail_loading_blocked():
            self._schedule_visible_thumbnail_loads()
            if self._frame_matrix_thumbnails_enabled() and hasattr(self, "_resume_thumbnail_radial_fill"):
                self._resume_thumbnail_radial_fill()

    def _schedule_thumbnail_grid_rebuild(self: Any, *, force: bool = False) -> None:
        if not self._frame_matrix_enabled():
            self._disable_frame_matrix_runtime()
            return
        if force and not getattr(self, "_frame_load_running_path", None) and not getattr(self, "_loading_image_path", None):
            self._pending_thumbnail_rebuild_after_vectors = False
            self._thumbnail_flush_retry_count = 0
            QTimer.singleShot(0, self._rebuild_thumbnail_grid)
            return
        self._mark_thumbnail_grid_rebuild_pending()
        self._flush_pending_thumbnail_grid_rebuild()

    def _load_cif_overlay_polygons(self: Any, image_path: str) -> list[PolygonData]:
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

    def load_image(self: Any, path: str, *, load_vectors: bool | None = None) -> None:
        normalized_load_path = str(Path(path))
        if load_vectors is None:
            load_vectors = not self._should_defer_vector_load()
        active_load_path = getattr(self, "_loading_image_path", None)
        if active_load_path is not None:
            if active_load_path == normalized_load_path:
                return
            self._frame_load_pending = (normalized_load_path, bool(load_vectors))
            return
        running_path = getattr(self, "_frame_load_running_path", None)
        if running_path is not None:
            self._frame_load_pending = (normalized_load_path, bool(load_vectors))
            return
        self._loading_image_path = normalized_load_path
        if hasattr(self, "_pause_thumbnail_radial_fill"):
            self._pause_thumbnail_radial_fill()
        try:
            self._begin_frame_load(normalized_load_path, load_vectors=bool(load_vectors))
        except Exception:
            if getattr(self, "_loading_image_path", None) == normalized_load_path:
                self._loading_image_path = None
            raise

    def _begin_frame_load(self: Any, normalized_load_path: str, *, load_vectors: bool) -> None:
        session = self._frame_switch_profile_for_path(normalized_load_path)
        if session is None:
            session = self._start_frame_switch_profile(normalized_load_path)
        if session is not None:
            self._append_log(
                f"[contour frame switch profiling] started image={Path(normalized_load_path).name} "
                "(runs until UI is interactive; set CONTOUR_PROFILE=0 or CONTOUR_PROFILE_FRAME_SWITCH=0 to disable)"
            )
        cif_path_for_profile = self._find_matching_cif_path(normalized_load_path)
        profile_timings: dict[str, float] = {}
        profile_total_start = perf_counter()
        try:
            phase_start = perf_counter()
            self._abort_in_flight_interactive_processing(preview=True, prepared=True)
            profile_timings["abort_in_flight"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()

            sync_result = self._workspace.resolve_cached_load(normalized_load_path)
            if sync_result is not None:
                state = sync_result.state
                needs_vectors = (
                    load_vectors
                    and state is not None
                    and not state.polygons
                    and bool(self._find_matching_cif_path(normalized_load_path))
                )
                if needs_vectors:
                    self._loading_image_path = normalized_load_path
                    self._begin_frame_vectors_reload(normalized_load_path)
                    return
                session = getattr(self, "_frame_switch_profile", None)
                if session is not None:
                    session.enable_main_profiler()
                self._finish_frame_load_ui(
                    sync_result,
                    load_vectors=load_vectors,
                    profile_enabled=session is not None,
                    profile_timings=profile_timings,
                    profile_total_start=profile_total_start,
                    profiler=session.profiler if session is not None else None,
                    cif_path_for_profile=cif_path_for_profile,
                    phase_start=perf_counter(),
                )
                return

            self._frame_load_running_path = normalized_load_path
            self._frame_load_request_serial = int(getattr(self, "_frame_load_request_serial", 0)) + 1
            request_id = self._frame_load_request_serial

            def load_source_image_timed(image_path: str):
                inner_start = perf_counter()
                session = getattr(self, "_frame_switch_profile", None)

                def _load() -> object:
                    return load_image_color(image_path)

                try:
                    return profile_callable("worker_source_image", session, _load)
                finally:
                    profile_timings["source_image_load"] = (perf_counter() - inner_start) * 1000.0

            def load_cif_overlay_timed(image_path: str) -> list[PolygonData]:
                inner_start = perf_counter()
                session = getattr(self, "_frame_switch_profile", None)

                def _load() -> list[PolygonData]:
                    return self._load_cif_overlay_polygons(image_path)

                try:
                    return profile_callable("worker_cif_overlay", session, _load)
                finally:
                    profile_timings["cif_overlay_load"] = (perf_counter() - inner_start) * 1000.0

            runnable = FrameLoadRunnable(
                request_id,
                normalized_load_path,
                load_source_image=load_source_image_timed,
                load_cif_overlay=load_cif_overlay_timed,
                load_vectors=load_vectors,
                vectors_only=False,
            )

            def _on_result(req_id: int, payload_obj: object) -> None:
                if bool(getattr(self, "_closing", False)):
                    return
                if req_id != self._frame_load_request_serial:
                    return
                payload = payload_obj
                if not isinstance(payload, FrameLoadPayload):
                    return
                session = self._frame_switch_profile_for_path(payload.image_path)
                if session is not None:
                    session.enable_main_profiler()
                apply_start = perf_counter()
                image_result = self._workspace.apply_loaded_frame(
                    payload.image_path,
                    source_image=payload.source_image,
                    polygons=list(payload.polygons),
                )
                profile_timings["workspace_apply_loaded_frame"] = (perf_counter() - apply_start) * 1000.0
                self._frame_load_running_path = None
                self._finish_frame_load_ui(
                    image_result,
                    load_vectors=load_vectors,
                    profile_enabled=session is not None,
                    profile_timings=profile_timings,
                    profile_total_start=profile_total_start,
                    profiler=session.profiler if session is not None else None,
                    cif_path_for_profile=cif_path_for_profile,
                    phase_start=perf_counter(),
                )
                if getattr(self, "_loading_image_path", None) == payload.image_path:
                    self._loading_image_path = None
                self._resume_frame_matrix_thumbnail_loading()
                self._flush_pending_thumbnail_grid_rebuild()
                self._drain_pending_frame_load()

            def _on_error(req_id: int, image_path: str, message: str) -> None:
                if bool(getattr(self, "_closing", False)):
                    return
                if req_id != self._frame_load_request_serial:
                    return
                self._frame_load_running_path = None
                if getattr(self, "_loading_image_path", None) == image_path:
                    self._loading_image_path = None
                self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=message))
                failed_session = getattr(self, "_frame_switch_profile", None)
                if failed_session is not None:
                    failed_session.disable_main_profiler()
                    self._frame_switch_profile = None
                QMessageBox.warning(self, self._tr("image_load_error_title"), message)
                self._resume_frame_matrix_thumbnail_loading()
                self._flush_pending_thumbnail_grid_rebuild()
                self._drain_pending_frame_load()

            runnable.signals.result.connect(_on_result)
            runnable.signals.error.connect(_on_error)
            session = getattr(self, "_frame_switch_profile", None)
            if session is not None:
                session.disable_main_profiler()
            self._frame_load_thread_pool.start(runnable)
        finally:
            pass

    def _reload_current_frame_vectors(self: Any) -> None:
        current = self._workspace.current_image_path
        if not current:
            return
        normalized = str(Path(current))
        state = self._workspace.current_state
        if state is None or state.source_image is None:
            self.load_image(normalized, load_vectors=True)
            return
        running_path = getattr(self, "_frame_load_running_path", None)
        if running_path is not None or getattr(self, "_loading_image_path", None) is not None:
            self._mark_thumbnail_grid_rebuild_pending()
            self._frame_load_pending = (normalized, True)
            return
        self._loading_image_path = normalized
        try:
            self._begin_frame_vectors_reload(normalized)
        except Exception:
            if getattr(self, "_loading_image_path", None) == normalized:
                self._loading_image_path = None
            raise

    def _begin_frame_vectors_reload(self: Any, normalized_load_path: str) -> None:
        if self._frame_switch_profile_for_path(normalized_load_path) is None:
            self._start_frame_switch_profile(normalized_load_path)
        cif_path_for_profile = self._find_matching_cif_path(normalized_load_path)
        profile_timings: dict[str, float] = {}
        profile_total_start = perf_counter()
        self._frame_load_running_path = normalized_load_path
        self._frame_load_request_serial = int(getattr(self, "_frame_load_request_serial", 0)) + 1
        request_id = self._frame_load_request_serial

        def load_cif_overlay_timed(image_path: str) -> list[PolygonData]:
            inner_start = perf_counter()
            session = getattr(self, "_frame_switch_profile", None)

            def _load() -> list[PolygonData]:
                return self._load_cif_overlay_polygons(image_path)

            try:
                return profile_callable("worker_cif_overlay", session, _load)
            finally:
                profile_timings["cif_overlay_load"] = (perf_counter() - inner_start) * 1000.0

        runnable = FrameLoadRunnable(
            request_id,
            normalized_load_path,
            load_source_image=None,
            load_cif_overlay=load_cif_overlay_timed,
            load_vectors=True,
            vectors_only=True,
        )

        def _on_vectors(req_id: int, payload_obj: object) -> None:
            if bool(getattr(self, "_closing", False)):
                return
            if req_id != self._frame_load_request_serial:
                return
            payload = payload_obj
            if not isinstance(payload, FrameLoadPayload):
                return
            session = self._frame_switch_profile_for_path(payload.image_path)
            if session is not None:
                session.enable_main_profiler()
            apply_start = perf_counter()
            image_result = self._workspace.apply_frame_vectors(
                payload.image_path,
                polygons=list(payload.polygons),
                loaded_cif_path=self._find_matching_cif_path(payload.image_path),
            )
            profile_timings["workspace_apply_frame_vectors"] = (perf_counter() - apply_start) * 1000.0
            self._frame_load_running_path = None
            if image_result is not None:
                self._finish_frame_load_ui(
                    image_result,
                    load_vectors=True,
                    profile_enabled=session is not None,
                    profile_timings=profile_timings,
                    profile_total_start=profile_total_start,
                    profiler=session.profiler if session is not None else None,
                    cif_path_for_profile=cif_path_for_profile,
                    phase_start=perf_counter(),
                )
            else:
                self.load_image(payload.image_path, load_vectors=True)
            if getattr(self, "_loading_image_path", None) == payload.image_path:
                self._loading_image_path = None
            self._flush_pending_thumbnail_grid_rebuild()
            self._drain_pending_frame_load()

        def _on_vectors_error(req_id: int, image_path: str, message: str) -> None:
            if bool(getattr(self, "_closing", False)):
                return
            if req_id != self._frame_load_request_serial:
                return
            self._frame_load_running_path = None
            if getattr(self, "_loading_image_path", None) == image_path:
                self._loading_image_path = None
            self._append_log(self._tr("reload_with_cif_failed_log", error=message))
            profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
            self._emit_cif_open_profile(
                profile_timings,
                image_path=image_path,
                cif_path=cif_path_for_profile,
                polygon_count=0,
                profiler=None,
                vectors_only=True,
                failed=True,
            )
            self._flush_pending_thumbnail_grid_rebuild()
            self._drain_pending_frame_load()

        runnable.signals.result.connect(_on_vectors)
        runnable.signals.error.connect(_on_vectors_error)
        session = getattr(self, "_frame_switch_profile", None)
        if session is not None:
            session.disable_main_profiler()
        self._frame_load_thread_pool.start(runnable)

    def _drain_pending_frame_load(self: Any) -> None:
        pending = getattr(self, "_frame_load_pending", None)
        if pending:
            self._frame_load_pending = None
            path, load_vectors = pending
            self.load_image(path, load_vectors=load_vectors)
            return
        self._flush_pending_thumbnail_grid_rebuild()

    def _finish_frame_load_ui(
        self: Any,
        image_result: WorkspaceLoadResult,
        *,
        load_vectors: bool,
        profile_enabled: bool = False,
        profile_timings: dict[str, float] | None = None,
        profile_total_start: float = 0.0,
        profiler: cProfile.Profile | None = None,
        cif_path_for_profile: str | None = None,
        phase_start: float = 0.0,
    ) -> None:
        profile_timings = profile_timings or {}
        phase_start = phase_start or perf_counter()
        self._save_persisted_current_image_path(image_result.image_path)
        if not self._is_extraction_mode_enabled():
            self._viewed_image_paths.add(str(Path(image_result.image_path)))
            self._handle_gamification_ui_event(RewardEventType.IMAGE_VIEWED)
        if (
            image_result.state is not None
            and not image_result.cache_hit
            and not image_result.reused_current_state
            and not image_result.vectors_only
        ):
            image_result.state.loaded_cif_path = self._find_matching_cif_path(image_result.image_path)
            image_result.state.reference_polygons = [polygon.clone() for polygon in image_result.state.polygons]
            image_result.state.polygons_dirty = False
        elif image_result.state is not None and image_result.vectors_only:
            image_result.state.loaded_cif_path = self._find_matching_cif_path(image_result.image_path)
            image_result.state.reference_polygons = [polygon.clone() for polygon in image_result.state.polygons]
            image_result.state.polygons_dirty = False
        if image_result.reused_current_state:
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self.polygon_editor.center_main_image()
            if profile_enabled:
                profile_timings["center_main_image"] = (perf_counter() - step_start) * 1000.0
            if getattr(self, "_loading_image_path", None) == image_result.image_path:
                self._loading_image_path = None
            step_start = perf_counter()
            self._resume_frame_matrix_thumbnail_loading()
            if profile_enabled:
                profile_timings["resume_frame_matrix_thumbnails"] = (perf_counter() - step_start) * 1000.0
            if profile_enabled:
                profile_timings["sync_reused"] = (perf_counter() - phase_start) * 1000.0
                self._emit_cif_open_profile(
                    profile_timings,
                    image_path=image_result.image_path,
                    cif_path=cif_path_for_profile,
                    polygon_count=0 if image_result.state is None else len(image_result.state.polygons),
                    profiler=profiler,
                    vectors_only=image_result.vectors_only,
                )
            return
        if image_result.vectors_only:
            current_state = self._workspace.current_state
            if current_state is not None:
                step_start = perf_counter()
                self._sync_polygons_to_editor(image_result.image_path, list(current_state.polygons))
                if profile_enabled:
                    profile_timings["editor_set_polygons"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._update_vector_edit_status_label()
            if profile_enabled:
                profile_timings["update_vector_status"] = (perf_counter() - step_start) * 1000.0
        elif image_result.cache_hit:
            self._updating_views = True
            try:
                step_start = perf_counter()
                self._apply_frame_to_editor(defer_neighbors=True, defer_heavy_overlays=False)
                if profile_enabled:
                    profile_timings["apply_frame_to_editor"] = (perf_counter() - step_start) * 1000.0
            finally:
                self._updating_views = False
            step_start = perf_counter()
            self._request_neighbor_frame_sync(delay_ms=0)
            if profile_enabled:
                profile_timings["sync_neighbor_frames"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
        else:
            self._updating_views = True
            try:
                step_start = perf_counter()
                self._apply_frame_to_editor(defer_neighbors=True, defer_heavy_overlays=False)
                if profile_enabled:
                    profile_timings["apply_frame_to_editor"] = (perf_counter() - step_start) * 1000.0
            finally:
                self._updating_views = False
            step_start = perf_counter()
            self._request_neighbor_frame_sync(delay_ms=0)
            if profile_enabled:
                profile_timings["sync_neighbor_frames"] = (perf_counter() - step_start) * 1000.0
            step_start = perf_counter()
            self._defer_frame_chrome_updates(image_result.image_path)
            if profile_enabled:
                profile_timings["defer_frame_chrome"] = (perf_counter() - step_start) * 1000.0
        if getattr(self, "_loading_image_path", None) == image_result.image_path:
            self._loading_image_path = None
        step_start = perf_counter()
        self._resume_frame_matrix_thumbnail_loading()
        if profile_enabled:
            profile_timings["resume_frame_matrix_thumbnails"] = (perf_counter() - step_start) * 1000.0
        if bool(getattr(self, "_pending_thumbnail_rebuild_after_vectors", False)):
            step_start = perf_counter()
            self._flush_pending_thumbnail_grid_rebuild()
            if profile_enabled:
                profile_timings["flush_pending_thumbnail_rebuild"] = (perf_counter() - step_start) * 1000.0
        if profile_enabled:
            profile_timings["sync_views"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
        if (
            image_result.prepared_image_required
            and image_result.state is not None
            and image_result.state.source_image is not None
            and not image_result.vectors_only
        ):
            self._queue_prepared_image_update(image_result.image_path, image_result.state.source_image)
        if profile_enabled:
            profile_timings["queue_prepared"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
        if image_result.cache_hit and not image_result.vectors_only:
            self._append_log(self._tr("loaded_cached_state_log", image_path=image_result.image_path))
        elif not image_result.vectors_only:
            self._append_log(self._tr("loaded_image_log", image_path=image_result.image_path))
        if profile_enabled:
            profile_timings["log"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
        if load_vectors and not image_result.vectors_only:
            profile_session = self._frame_switch_profile_for_path(image_result.image_path)
            if profile_session is not None:
                profile_session.disable_main_profiler()
            self._try_extract_if_recognition_enabled()
        if profile_enabled:
            profile_timings["maybe_extract"] = (perf_counter() - phase_start) * 1000.0
            profile_session = self._frame_switch_profile_for_path(image_result.image_path)
            if profile_session is not None:
                profile_session.enable_main_profiler()
            self._emit_cif_open_profile(
                profile_timings,
                image_path=image_result.image_path,
                cif_path=cif_path_for_profile,
                polygon_count=0 if image_result.state is None else len(image_result.state.polygons),
                profiler=profiler,
                vectors_only=image_result.vectors_only,
            )

    def _frame_switch_profiling_active(self: Any) -> bool:
        return frame_switch_profiling_enabled()

    def _start_frame_switch_profile(self: Any, image_path: str) -> FrameSwitchProfile | None:
        if not self._frame_switch_profiling_active():
            return None
        self._frame_switch_profile_generation = int(getattr(self, "_frame_switch_profile_generation", 0)) + 1
        session = FrameSwitchProfile.begin(
            image_path,
            generation=self._frame_switch_profile_generation,
        )
        self._frame_switch_profile = session
        return session

    def _frame_switch_profile_for_path(self: Any, image_path: str) -> FrameSwitchProfile | None:
        session = getattr(self, "_frame_switch_profile", None)
        if session is None:
            return None
        if str(Path(session.image_path)) != str(Path(image_path)):
            return None
        return session

    def _frame_switch_profile_is_interactive(self: Any, image_path: str) -> bool:
        normalized = str(Path(image_path))
        if str(Path(self._workspace.current_image_path or "")) != normalized:
            return False
        if getattr(self, "_loading_image_path", None) is not None:
            return False
        if getattr(self, "_frame_load_running_path", None) is not None:
            return False
        if getattr(self, "_pending_editor_frame_apply", None) is not None:
            return False
        if getattr(self, "_thumbnail_rebuild_in_progress", False):
            return False
        if self._editor_display_thread_pool.activeThreadCount() > 0:
            return False
        if self._frame_load_thread_pool.activeThreadCount() > 0:
            return False
        chrome_timer = getattr(self, "_frame_chrome_update_timer", None)
        if chrome_timer is not None and chrome_timer.isActive():
            return False
        neighbor_timer = getattr(self, "_neighbor_sync_timer", None)
        if neighbor_timer is not None and neighbor_timer.isActive():
            return False
        session = self._frame_switch_profile_for_path(normalized)
        if session is not None and session.pending_since:
            return False
        return True

    def _schedule_frame_switch_profile_until_interactive(
        self: Any,
        image_path: str,
        *,
        cif_path: str | None,
        polygon_count: int,
        vectors_only: bool = False,
        failed: bool = False,
    ) -> None:
        session = self._frame_switch_profile_for_path(image_path)
        if session is None:
            return
        generation = session.generation
        QTimer.singleShot(
            0,
            lambda: self._poll_frame_switch_profile_until_interactive(
                image_path,
                generation,
                cif_path=cif_path,
                polygon_count=polygon_count,
                vectors_only=vectors_only,
                failed=failed,
            ),
        )

    def _poll_frame_switch_profile_until_interactive(
        self: Any,
        image_path: str,
        generation: int,
        *,
        cif_path: str | None,
        polygon_count: int,
        vectors_only: bool,
        failed: bool,
    ) -> None:
        session = getattr(self, "_frame_switch_profile", None)
        if session is None or session.generation != generation:
            return
        session.poll_count += 1
        interactive = failed or self._frame_switch_profile_is_interactive(image_path)
        if not interactive and session.poll_count < MAX_IDLE_POLLS:
            QTimer.singleShot(
                0,
                lambda: self._poll_frame_switch_profile_until_interactive(
                    image_path,
                    generation,
                    cif_path=cif_path,
                    polygon_count=polygon_count,
                    vectors_only=vectors_only,
                    failed=failed,
                ),
            )
            return
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self._finalize_frame_switch_profile(
            session,
            cif_path=cif_path,
            polygon_count=polygon_count,
            vectors_only=vectors_only,
            failed=failed,
            interactive=interactive and not failed,
        )

    def _finalize_frame_switch_profile(
        self: Any,
        session: FrameSwitchProfile,
        *,
        cif_path: str | None,
        polygon_count: int,
        vectors_only: bool,
        failed: bool,
        interactive: bool,
    ) -> None:
        if getattr(self, "_frame_switch_profile", None) is session:
            self._frame_switch_profile = None
        session.timings_ms["total_wall"] = session.total_wall_ms()
        main_profiler = session.disable_main_profiler()
        summary = session.format_summary(
            polygon_count=polygon_count,
            cif_path=cif_path,
            vectors_only=vectors_only,
            failed=failed,
            interactive=interactive,
        )
        print(summary)
        self._append_log(summary)
        if session.profiling_active or main_profiler.getstats():
            print(session.format_stats(main_profiler, title="main_thread_until_interactive"))
        elif session.main_stats_skipped:
            print(
                "[contour frame switch profiling stats] main_thread_until_interactive skipped "
                "(another cProfile session was active, e.g. contour processing extract)"
            )
        for label, worker_profiler in session.worker_profilers:
            print(session.format_stats(worker_profiler, title=f"worker_{label}"))
        self._append_log(
            "[contour frame switch profiling stats] printed to console "
            "(main thread until interactive; see worker_* sections for background load)"
        )
        QTimer.singleShot(100, self._resume_frame_matrix_thumbnail_loading)

    def _emit_cif_open_profile(
        self,
        timings_ms: dict[str, float],
        *,
        image_path: str,
        cif_path: str | None,
        polygon_count: int,
        profiler: cProfile.Profile | None,
        vectors_only: bool = False,
        failed: bool = False,
    ) -> None:
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.merge_timings(timings_ms)
            if profiler is not None and profiler is not session.profiler:
                session.attach_worker_profile("ui_finish", profiler)
            self._schedule_frame_switch_profile_until_interactive(
                image_path,
                cif_path=cif_path,
                polygon_count=polygon_count,
                vectors_only=vectors_only,
                failed=failed,
            )
            return
        total_ms = timings_ms.get("total_wall", sum(timings_ms.values()))
        detail = " ".join(
            f"{name}={elapsed:.3f}ms" for name, elapsed in timings_ms.items() if name != "total_wall"
        )
        mode = "vector" if vectors_only else "image+vector"
        status = "failed" if failed else "ok"
        message = (
            f"[contour frame open profiling] mode={mode} status={status} total={total_ms:.3f}ms "
            f"polygons={polygon_count} image={Path(image_path).name} "
            f"cif={Path(cif_path).name if cif_path else '<none>'} {detail}"
        )
        print(message)
        self._append_log(message)

    def _set_work_simulation_running(self: Any, running: bool) -> None:
        running = bool(running)
        if getattr(self, "_work_simulation_running", False) == running:
            return
        self._work_simulation_running = running
        if hasattr(self, "workSimulationActiveChanged"):
            self.workSimulationActiveChanged.emit(running)

    def _toggle_work_simulation(self: Any) -> None:
        if getattr(self, "_work_simulation_running", False):
            self._stop_work_simulation(restore_current=True)
            self._append_log("Симуляция остановлена." if self._ui_language == "ru" else "Work simulation stopped.")
            return
        self._start_work_simulation()

    def _start_work_simulation(self: Any) -> None:
        self._stop_work_simulation(restore_current=True)
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        paths = [
            str(Path(path))
            for path in self._workspace.image_paths
            if self._workspace.resolve_cif_path(str(Path(path)))
        ]
        if not paths:
            self._append_log(
                "Нет кадров с изображением и вектором для симуляции."
                if self._ui_language == "ru"
                else "No image+vector frames are available for simulation."
            )
            return
        self._set_work_simulation_running(True)
        self._work_simulation_paths = paths
        self._work_simulation_path_index = -1
        self._work_simulation_target_polygons = []
        self._work_simulation_visible_points = 0
        self._work_simulation_total_points = 0
        self._work_simulation_timer.setInterval(max(1, int(self._work_simulation_interval_ms)))
        self._advance_work_simulation()

    def _stop_work_simulation(self: Any, *, restore_current: bool = True) -> None:
        if hasattr(self, "_work_simulation_timer"):
            self._work_simulation_timer.stop()
        if restore_current and getattr(self, "_work_simulation_target_polygons", None):
            self._restore_work_simulation_frame()
        self._work_simulation_paths = []
        self._work_simulation_path_index = -1
        self._work_simulation_target_polygons = []
        self._work_simulation_visible_points = 0
        self._work_simulation_total_points = 0
        self._work_simulation_original_dirty = None
        self._work_simulation_original_reference_polygons = []
        self._set_work_simulation_running(False)

    def _advance_work_simulation(self: Any) -> None:
        if not getattr(self, "_work_simulation_running", False):
            return
        if not getattr(self, "_work_simulation_target_polygons", None):
            self._begin_next_work_simulation_frame()
            return
        self._work_simulation_visible_points += 1
        if self._work_simulation_visible_points >= self._work_simulation_total_points:
            self._work_simulation_timer.stop()
            self._restore_work_simulation_frame()
            self._work_simulation_target_polygons = []
            QTimer.singleShot(max(1, int(self._work_simulation_interval_ms)), self._begin_next_work_simulation_frame)
            return
        partial = self._partial_work_simulation_polygons(
            self._work_simulation_target_polygons,
            self._work_simulation_visible_points,
        )
        self._set_editor_polygons_for_work_simulation(partial)
        self._work_simulation_timer.start(max(1, int(self._work_simulation_interval_ms)))

    def _begin_next_work_simulation_frame(self: Any) -> None:
        if not getattr(self, "_work_simulation_running", False):
            return
        self._work_simulation_timer.stop()
        self._work_simulation_path_index += 1
        if self._work_simulation_path_index >= len(self._work_simulation_paths):
            self._stop_work_simulation(restore_current=False)
            self._append_log("Симуляция завершена." if self._ui_language == "ru" else "Work simulation finished.")
            return
        path = self._work_simulation_paths[self._work_simulation_path_index]
        if self._workspace.current_image_path != path:
            if self._image_path_in_image_list(path):
                self._set_image_list_current_path(path, fallback_to_first=False)
            else:
                self.load_image(path)
        state = self._workspace.current_state
        if state is None or self._workspace.current_image_path != path:
            QTimer.singleShot(0, self._begin_next_work_simulation_frame)
            return
        target = [polygon.clone() for polygon in state.polygons]
        if not target:
            QTimer.singleShot(0, self._begin_next_work_simulation_frame)
            return
        self._work_simulation_target_polygons = target
        self._work_simulation_visible_points = 0
        self._work_simulation_total_points = max(1, sum(len(polygon.points) for polygon in target))
        self._work_simulation_original_dirty = state.polygons_dirty
        self._work_simulation_original_reference_polygons = [polygon.clone() for polygon in state.reference_polygons]
        self._set_editor_polygons_for_work_simulation([])
        self._work_simulation_timer.start(max(1, int(self._work_simulation_interval_ms)))

    def _set_editor_polygons_for_work_simulation(self: Any, polygons: list[PolygonData]) -> None:
        self._updating_views = True
        try:
            self.polygon_editor.set_polygons([polygon.clone() for polygon in polygons])
        finally:
            self._updating_views = False

    def _partial_work_simulation_polygons(
        self,
        polygons: list[PolygonData],
        visible_points: int,
    ) -> list[PolygonData]:
        remaining = max(0, int(visible_points))
        partial: list[PolygonData] = []
        for polygon in polygons:
            if remaining <= 0:
                break
            point_count = min(len(polygon.points), remaining)
            if point_count > 0:
                clone = polygon.clone()
                clone.points = list(clone.points[:point_count])
                partial.append(clone)
            remaining -= len(polygon.points)
        return partial

    def _restore_work_simulation_frame(self: Any) -> None:
        target = [polygon.clone() for polygon in getattr(self, "_work_simulation_target_polygons", [])]
        if not target:
            return
        state = self._workspace.current_state
        if state is not None:
            state.polygons = [polygon.clone() for polygon in target]
            state.reference_polygons = [
                polygon.clone() for polygon in getattr(self, "_work_simulation_original_reference_polygons", [])
            ]
            state.polygons_dirty = getattr(self, "_work_simulation_original_dirty", None)
        self._set_editor_polygons_for_work_simulation(target)
        self._update_frame_item_status(self._workspace.current_image_path)
        self._update_vector_edit_status_label()

    def _is_extraction_mode_enabled(self: Any) -> bool:
        if not hasattr(self, "recognition_mode_combo"):
            return False
        return str(self.recognition_mode_combo.currentData() or "") != "disabled"

    def get_polygons(self: Any) -> list[PolygonData]:
        return self.polygon_editor.get_polygons()

    def set_pipeline(self: Any, config: dict) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(config)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def get_pipeline(self: Any) -> dict:
        return self._pipeline.to_dict()

    def process_current_image(self: Any, *_args, debounced: bool = False) -> None:
        self._queue_preview_processing(debounced=debounced)

    def _export_dataset_frame_for_state(
        self: Any,
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

    def export_current_frame_to_dataset(self: Any, dataset_directory: str | None = None) -> dict[str, str]:
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
        self: Any,
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
        self: Any,
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
                max_workers=mp.cpu_count(),
                chunk_size=max(1, int(getattr(self, "batch_chunk_size", 16))),
                )
            )
        if not started:
            return
        self._batch_progress_enabled = self._batch_controller.progress_enabled
        self._show_batch_progress(len(paths))
        self._set_progress_status("batch_started_status")

    def stop_batch_processing(self: Any) -> None:
        self._batch_controller.stop()

    def _handle_gamification_after_save(
        self: Any,
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

    def _handle_gamification_ui_event(self: Any, event_type: RewardEventType) -> None:
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
