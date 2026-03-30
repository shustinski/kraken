"""Coordinate mismatch-only UI events, persistence and worker lifecycle for the lite widget."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, QSignalBlocker, QThread, Qt
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMenu, QMessageBox

from ..core.domain import BuildOptions, BuildResult, ComparisonMode, FolderSpec, FrameIdentity, FrameRecord
from ..ui.matrix_view import MatrixLayoutConfig, build_matrix_layout
from ..ui.ui_components import FolderRowWidget
from ..core.workers import FrameIndexWorker, MismatchWorker

from ..ui.details_dialog import LiteFrameDetailsDialog
from ..infra.services import ValidationGradientLiteSettingsService
from .state import LiteMatrixTabState
from ..ui.ui_constants import (
    DEFAULT_COMPARISON_MODE,
    DEFAULT_ERROR_WINDOW,
    DEFAULT_FRAMES_PER_ROW,
    DEFAULT_MATRIX_COLUMNS,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_ROWS,
    DEFAULT_GRADIENT_NAME,
    DEFAULT_SCORE_VIEW_MODE,
    DEFAULT_TOTAL_FRAMES,
    FOLDER_CHECKED_ROLE,
    FOLDER_LABEL_ROLE,
    FOLDER_ROW_MIN_HEIGHT,
    REQUIRED_COMPARE_FOLDER_COUNT,
)


class ValidationGradientLitePresenter(QObject):
    """Coordinate widget state, persistence and mismatch workers for the lite widget."""

    def __init__(self, view, settings_service: ValidationGradientLiteSettingsService) -> None:
        super().__init__(view)
        self._view = view
        self._settings_service = settings_service
        self._worker_thread: QThread | None = None
        self._worker = None
        self._worker_kind: str | None = None
        self._active_compute_state: LiteMatrixTabState | None = None
        self._base_folder: FolderSpec | None = None
        self._folder_check_guard = False
        self._tab_states: dict[object, LiteMatrixTabState] = {}
        self._pending_build_snapshot: dict[str, object] | None = None

    def __getattr__(self, name: str):
        return getattr(self._view, name)

    def _current_tab_state(self) -> LiteMatrixTabState | None:
        widget = self.matrix_tabs.currentWidget()
        if widget is None:
            return None
        return self._tab_states.get(widget)

    def _checked_folders(self) -> list[FolderSpec]:
        folders: list[FolderSpec] = []
        for row in range(self.folder_list.count()):
            item = self.folder_list.item(row)
            if bool(item.data(FOLDER_CHECKED_ROLE)):
                folder_path = Path(item.data(Qt.ItemDataRole.UserRole))
                folder_label = str(item.data(FOLDER_LABEL_ROLE) or folder_path.name)
                folders.append(FolderSpec(path=folder_path, label=folder_label))
        return folders

    def _append_folder_item(self, folder_path: Path, *, checked: bool) -> QListWidgetItem:
        folder_path = Path(folder_path)
        for row in range(self.folder_list.count()):
            existing_item = self.folder_list.item(row)
            if Path(existing_item.data(Qt.ItemDataRole.UserRole)) == folder_path:
                existing_item.setData(FOLDER_CHECKED_ROLE, bool(checked))
                if not existing_item.data(FOLDER_LABEL_ROLE):
                    existing_item.setData(FOLDER_LABEL_ROLE, folder_path.name)
                return existing_item
        item = QListWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.ItemDataRole.UserRole, str(folder_path))
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        item.setData(FOLDER_LABEL_ROLE, folder_path.name)
        item.setToolTip(str(folder_path))
        self.folder_list.addItem(item)
        return item

    def _refresh_folder_rows(self) -> None:
        for row in range(self.folder_list.count()):
            item = self.folder_list.item(row)
            path_text = str(item.data(Qt.ItemDataRole.UserRole))
            display_text = str(item.data(FOLDER_LABEL_ROLE) or (Path(path_text).name or path_text))
            row_widget = FolderRowWidget(
                self.folder_list,
                path_text=path_text,
                display_text=display_text,
                checked=bool(item.data(FOLDER_CHECKED_ROLE)),
                can_move_up=row > 0,
                can_move_down=row < self.folder_list.count() - 1,
                on_checked_changed=lambda checked, item=item: self._set_folder_item_checked(item, checked),
                on_label_changed=lambda text, item=item: self._set_folder_item_label(item, text),
                on_remove=lambda _checked=False, item=item: self._remove_folder_item(item),
                on_move_up=lambda _checked=False, item=item: self._move_folder_item(item, -1),
                on_move_down=lambda _checked=False, item=item: self._move_folder_item(item, 1),
                checkbox_tooltip=self._t("folders.use_compare"),
                remove_tooltip=self._t("folders.remove"),
                move_up_tooltip=self._t("folders.move_up"),
                move_down_tooltip=self._t("folders.move_down"),
            )
            row_widget.setMinimumHeight(FOLDER_ROW_MIN_HEIGHT)
            item.setSizeHint(row_widget.sizeHint())
            self.folder_list.setItemWidget(item, row_widget)

    def _set_folder_item_checked(self, item: QListWidgetItem, checked: bool) -> None:
        self._folder_check_guard = True
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        self._folder_check_guard = False
        self._on_folder_item_changed(item)
        self._refresh_folder_rows()

    def _set_folder_item_label(self, item: QListWidgetItem, text: str) -> None:
        folder_path = Path(item.data(Qt.ItemDataRole.UserRole))
        item.setData(FOLDER_LABEL_ROLE, text or folder_path.name)
        self._refresh_folder_rows()

    def _remove_folder_item(self, item: QListWidgetItem) -> None:
        row = self.folder_list.row(item)
        if row < 0:
            return
        self.folder_list.takeItem(row)
        self._sync_action_buttons()
        self._refresh_folder_rows()

    def _move_folder_item(self, item: QListWidgetItem, delta: int) -> None:
        row = self.folder_list.row(item)
        target_row = row + int(delta)
        if row < 0 or target_row < 0 or target_row >= self.folder_list.count():
            return
        moved_item = self.folder_list.takeItem(row)
        self.folder_list.insertItem(target_row, moved_item)
        self.folder_list.setCurrentRow(target_row)
        self._refresh_folder_rows()

    def _build_layout_config(self) -> MatrixLayoutConfig:
        return MatrixLayoutConfig(
            mode=str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE),
            total_frames=int(self.total_frames_spin.value()),
            frames_per_row=int(self.frames_per_row_spin.value()),
            rows=int(self.matrix_rows_spin.value()),
            columns=int(self.matrix_columns_spin.value()),
        )

    def _capture_view_snapshot(self) -> dict[str, object]:
        return {
            "cell_size": int(self.thumbnail_size_spin.value()),
            "layout_config": self._build_layout_config(),
            "gradient_name": self.gradient_selector.selected_gradient(),
            "error_window": self.gradient_range_selector.error_window(),
            "score_view_mode": str(self.error_score_view_combo.currentData() or DEFAULT_SCORE_VIEW_MODE),
        }

    def _load_error_view_from_state(self, state: LiteMatrixTabState) -> None:
        self.gradient_selector.set_selected_gradient(state.gradient_name, emit_signal=False)
        self.gradient_range_selector.set_gradient_name(state.gradient_name)
        self.gradient_range_selector.set_error_window(*state.error_window)
        self._refresh_score_view_controls(state)
        self._update_matrix_preview(state)

    def _sync_layout_control_state(self) -> None:
        mode = str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE)
        is_indexed = mode != "manual_grid"
        self.total_frames_spin.setEnabled(is_indexed)
        self.frames_per_row_spin.setEnabled(is_indexed)
        self.matrix_rows_spin.setEnabled(not is_indexed)
        self.matrix_columns_spin.setEnabled(not is_indexed)
        for row_widget, visible in (
            (getattr(self, "_matrix_total_frames_row", None), is_indexed),
            (getattr(self, "_matrix_frames_per_row_row", None), is_indexed),
            (getattr(self, "_matrix_rows_row", None), not is_indexed),
            (getattr(self, "_matrix_columns_row", None), not is_indexed),
        ):
            if row_widget is not None:
                row_widget.setVisible(visible)

    def _attach_matrix_coordinates(self, state: LiteMatrixTabState) -> tuple[FrameRecord, ...]:
        placements, _columns, _rows = build_matrix_layout(list(state.build_result.records), state.layout_config)
        records_by_key: dict[str, FrameRecord] = {}
        for placement_index, (record, row, column) in enumerate(placements):
            identity = record.identity
            if identity is None:
                frame_id = placement_index
                identity = FrameIdentity(
                    frame_id=frame_id,
                    base_id=frame_id if record.base_path else None,
                    tile_x=column,
                    tile_y=row,
                    source_key=record.key,
                )
            else:
                identity = replace(identity, tile_x=column, tile_y=row)
            records_by_key[record.key] = replace(record, identity=identity)
        return tuple(records_by_key.get(record.key, record) for record in state.build_result.records)

    def _apply_tab_visual_settings(self, state: LiteMatrixTabState, *, reset_view: bool = False) -> bool:
        try:
            positioned_records = self._attach_matrix_coordinates(state)
            state.build_result = replace(state.build_result, records=positioned_records)
            state.matrix_view.set_gradient_preset(state.gradient_name)
            state.matrix_view.set_error_window(*state.error_window)
            state.matrix_view.set_cell_size(int(state.cell_size))
            state.matrix_view.set_layout_config(state.layout_config)
            state.matrix_view.set_reference_key(state.build_result.best_match_key)
            sort_mode = "input_order" if str(state.layout_config.mode or "indexed_grid") == "manual_grid" else "name"
            state.matrix_view.set_records(list(state.build_result.records), sort_mode=sort_mode, reset_view=reset_view)
            self._update_matrix_preview(state)
        except ValueError as error:
            QMessageBox.warning(self._view, self._t("errors.layout"), str(error))
            return False
        return True

    def _refresh_score_view_controls(self, state: LiteMatrixTabState | None = None) -> None:
        current_state = state or self._current_tab_state()
        self.error_score_view_combo.blockSignals(True)
        mode = current_state.score_view_mode if current_state is not None else DEFAULT_SCORE_VIEW_MODE
        index = self.error_score_view_combo.findData(mode)
        self.error_score_view_combo.setCurrentIndex(index if index >= 0 else 0)
        self.error_score_view_combo.setEnabled(current_state is not None and current_state.build_result.scores_computed)
        self.error_score_view_combo.blockSignals(False)

    def _apply_score_view_mode(self, state: LiteMatrixTabState, mode: str) -> None:
        state.score_view_mode = "absolute" if mode == "absolute" else "relative"
        self._refresh_score_view_controls(state)
        if not state.build_result.scores_computed:
            return
        updated_records: list[FrameRecord] = []
        for record in state.build_result.records:
            score_value = record.absolute_score if state.score_view_mode == "absolute" else record.relative_score
            if score_value is None:
                score_value = record.score
            updated_records.append(replace(record, score=float(score_value)))
        scores = [record.score for record in updated_records]
        state.build_result = replace(
            state.build_result,
            records=tuple(updated_records),
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
        )
        self._apply_tab_visual_settings(state, reset_view=False)

    def _update_matrix_preview(self, state: LiteMatrixTabState, record: FrameRecord | None = None) -> None:
        selected = record or state.matrix_view.current_record()
        frame_value = selected.display_name if selected is not None else "-"
        if selected is None or selected.absolute_score is None:
            absolute_value = self._t("matrix.not_computed")
        else:
            absolute_value = f"{selected.absolute_score:.4f}"
        if selected is None or selected.relative_score is None:
            relative_value = self._t("matrix.not_computed")
        else:
            relative_value = f"{selected.relative_score:.4f}"
        preview = state.preview
        if preview is None:
            return
        preview.frame_value.setText(frame_value)
        preview.absolute_value.setText(absolute_value)
        preview.relative_value.setText(relative_value)

    def _show_progress_bar(self, *, visible: bool, current: int = 0, total: int = 0, key: str = "", format_text: str | None = None) -> None:
        if not visible:
            self.build_progress.hide()
            self.build_progress.setRange(0, 1)
            self.build_progress.setValue(0)
            return
        if total > 0:
            self.build_progress.setRange(0, total)
            self.build_progress.setValue(min(current, total))
            self.build_progress.setFormat(format_text or self._t("progress.building", current=current, total=total))
        else:
            self.build_progress.setRange(0, 0)
            self.build_progress.setFormat(format_text or self._t("progress.starting"))
        self.build_progress.setToolTip(key)
        self.build_progress.show()

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_result_folder"))
        if not folder:
            return
        item = self._append_folder_item(Path(folder), checked=len(self._checked_folders()) < REQUIRED_COMPARE_FOLDER_COUNT)
        self.folder_list.setCurrentItem(item)
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _clear_folders(self) -> None:
        self.folder_list.clear()
        self._base_folder = None
        self._sync_action_buttons()
        self._refresh_folder_rows()
        self._show_progress_bar(visible=False)

    def _set_base_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_base_folder"))
        if not folder:
            return
        folder_path = Path(folder)
        self._base_folder = FolderSpec(path=folder_path, label=folder_path.name)
        self._sync_action_buttons()

    def _clear_base_folder(self) -> None:
        self._base_folder = None
        self._sync_action_buttons()

    def _on_folder_item_changed(self, item: QListWidgetItem) -> None:
        if self._folder_check_guard:
            return
        if not bool(item.data(FOLDER_CHECKED_ROLE)):
            self._sync_action_buttons()
            return
        checked_items = [
            self.folder_list.item(row)
            for row in range(self.folder_list.count())
            if bool(self.folder_list.item(row).data(FOLDER_CHECKED_ROLE))
        ]
        if len(checked_items) > REQUIRED_COMPARE_FOLDER_COUNT:
            self._folder_check_guard = True
            item.setData(FOLDER_CHECKED_ROLE, False)
            self._folder_check_guard = False
            QMessageBox.information(self._view, self._t("dialog.info_title"), self._t("message.max_two"))
        self._sync_action_buttons()
        self._refresh_folder_rows()

    def _start_build(self) -> None:
        compare_folders = self._checked_folders()
        if len(compare_folders) != REQUIRED_COMPARE_FOLDER_COUNT:
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), self._t("message.mark_exactly_two"))
            return
        options = BuildOptions(
            comparison_mode=self.mode_combo.currentData(),
            thumbnail_size=int(self.thumbnail_size_spin.value()),
            recursive=True,
        )
        self._pending_build_snapshot = self._capture_view_snapshot()
        self._show_progress_bar(visible=True, format_text=self._t("progress.indexing"))
        self._worker_kind = "build"
        self._active_compute_state = None
        self._worker_thread = QThread(self._view)
        self._worker = FrameIndexWorker(compare_folders[0], compare_folders[1], options, self._base_folder)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_build_progress)
        self._worker.finished.connect(self._on_build_finished)
        self._worker.failed.connect(self._on_build_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()
        self._sync_action_buttons()

    def _start_compute_mismatches(self) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        self._show_progress_bar(visible=True, format_text=self._t("progress.computing_start"))
        state.matrix_view.set_processing_keys(set())
        self._worker_kind = "compute"
        self._active_compute_state = state
        self._worker_thread = QThread(self._view)
        self._worker = MismatchWorker(state.build_result, self.mode_combo.currentData())
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_mismatch_progress)
        self._worker.activeKeysChanged.connect(self._on_processing_keys_changed)
        self._worker.finished.connect(self._on_mismatch_finished)
        self._worker.failed.connect(self._on_build_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()
        self._sync_action_buttons()

    def _request_cancel_build(self) -> None:
        if self._worker is None:
            return
        request_cancel = getattr(self._worker, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()
        self._show_progress_bar(visible=True, format_text=self._t("progress.cancelling"))

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        self._worker = None
        self._worker_thread = None
        self._worker_kind = None
        self._active_compute_state = None
        self._pending_build_snapshot = None
        self._sync_action_buttons()

    def _on_build_progress(self, current: int, total: int, key: str) -> None:
        if total > 0:
            self._show_progress_bar(
                visible=True,
                current=current,
                total=total,
                key=key,
                format_text=self._t("progress.building", current=current, total=total),
            )
        else:
            self._show_progress_bar(visible=True, key=key, format_text=self._t("progress.indexing"))

    def _on_mismatch_progress(self, current: int, total: int, key: str) -> None:
        if total > 0:
            self._show_progress_bar(
                visible=True,
                current=current,
                total=total,
                key=key,
                format_text=self._t("progress.computing", current=current, total=total),
            )
        else:
            self._show_progress_bar(visible=True, key=key, format_text=self._t("progress.computing_start"))

    def _on_processing_keys_changed(self, keys) -> None:
        state = self._active_compute_state or self._current_tab_state()
        if state is not None:
            state.matrix_view.set_processing_keys(set(keys))

    def _make_tab_title(self, build_result: BuildResult) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{build_result.first_folder.label} vs {build_result.second_folder.label} [{timestamp}]"

    def _on_build_finished(self, result: BuildResult) -> None:
        snapshot = self._pending_build_snapshot or self._capture_view_snapshot()
        state = self._create_matrix_tab(result, snapshot)
        ok = self._apply_tab_visual_settings(state, reset_view=True)
        self._show_progress_bar(visible=False)
        if not ok:
            state.widget.deleteLater()
            return
        tab_title = self._make_tab_title(result)
        self._tab_states[state.widget] = state
        tab_index = self.matrix_tabs.addTab(state.widget, tab_title)
        self.matrix_tabs.setCurrentIndex(tab_index)
        if result.records:
            state.matrix_view.select_record_by_key(result.records[0].key, ensure_visible=False)
            self._update_matrix_preview(state, result.records[0])
        self._sync_action_buttons()

    def _on_mismatch_finished(self, result: BuildResult) -> None:
        state = self._active_compute_state or self._current_tab_state()
        self._show_progress_bar(visible=False)
        if state is None:
            return
        state.build_result = result
        state.matrix_view.set_processing_keys(set())
        state.matrix_view.set_reference_key(result.best_match_key)
        self._apply_score_view_mode(state, state.score_view_mode)
        self._sync_action_buttons()

    def _on_build_failed(self, message: str) -> None:
        state = self._active_compute_state or self._current_tab_state()
        if state is not None:
            state.matrix_view.set_processing_keys(set())
        self._show_progress_bar(visible=False)
        if message and "cancel" in message.lower():
            self._sync_action_buttons()
            return
        QMessageBox.critical(self._view, self._t("dialog.warning_title"), message or self._t("message.build_failed"))

    def _on_comparison_mode_changed(self, *_args) -> None:
        self._sync_action_buttons()

    def _on_matrix_layout_mode_changed(self, *_args) -> None:
        self._sync_layout_control_state()
        self._on_matrix_visual_parameter_changed()

    def _on_matrix_visual_parameter_changed(self, *_args) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        state.cell_size = int(self.thumbnail_size_spin.value())
        state.layout_config = self._build_layout_config()
        self._apply_tab_visual_settings(state, reset_view=False)

    def _on_gradient_preset_changed(self, gradient_name: str) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        state.gradient_name = str(gradient_name)
        self.gradient_range_selector.set_gradient_name(state.gradient_name)
        self._apply_tab_visual_settings(state, reset_view=False)

    def _on_error_window_changed(self, low_bound: float, high_bound: float) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        state.error_window = (float(low_bound), float(high_bound))
        self._apply_tab_visual_settings(state, reset_view=False)

    def _on_error_score_view_changed(self, *_args) -> None:
        state = self._current_tab_state()
        if state is None:
            return
        self._apply_score_view_mode(state, str(self.error_score_view_combo.currentData() or DEFAULT_SCORE_VIEW_MODE))

    def _on_current_tab_changed(self, _index: int) -> None:
        state = self._current_tab_state()
        if state is not None:
            self._load_error_view_from_state(state)
        else:
            self._refresh_score_view_controls(None)
        self._sync_action_buttons()

    def _close_matrix_tab(self, index: int) -> None:
        widget = self.matrix_tabs.widget(index)
        if widget is None:
            return
        self._tab_states.pop(widget, None)
        self.matrix_tabs.removeTab(index)
        widget.deleteLater()
        self._sync_action_buttons()

    def _on_matrix_overview_changed(self, state: LiteMatrixTabState, image, visible_rect, selected_position, selected_blink_on, processing_positions, reference_position) -> None:
        state.mini_map.set_overview(
            image,
            visible_rect,
            selected_position,
            selected_blink_on,
            processing_positions,
            reference_position,
        )

    def _show_matrix_context_menu(self, state: LiteMatrixTabState, record: FrameRecord | None, global_pos) -> None:
        self.matrix_tabs.setCurrentWidget(state.widget)
        if record is None:
            return
        menu = QMenu(self._view)
        open_action = menu.addAction(self._t("context.open_details"))
        open_action.triggered.connect(lambda: self._open_record_details(record, state))
        menu.exec(global_pos)

    def _on_record_selected(self, state: LiteMatrixTabState, record: FrameRecord | None) -> None:
        if self._current_tab_state() is state:
            self._update_matrix_preview(state, record)
            self._sync_action_buttons()

    def _open_record_details(self, record: FrameRecord, state: LiteMatrixTabState) -> None:
        dialog = LiteFrameDetailsDialog(record=record, build_result=state.build_result, parent=self._view)
        dialog.exec()

    def _sync_action_buttons(self) -> None:
        current_state = self._current_tab_state()
        has_folders = self.folder_list.count() > 0
        checked_count = len(self._checked_folders())
        is_building = self._worker_thread is not None
        self.btn_clear_folders.setEnabled(has_folders and not is_building)
        self.btn_set_base.setEnabled(not is_building)
        self.btn_clear_base.setEnabled(self._base_folder is not None and not is_building)
        self.btn_build.setEnabled(checked_count == REQUIRED_COMPARE_FOLDER_COUNT and not is_building)
        self.btn_compute.setEnabled(current_state is not None and not is_building)
        self.btn_compute.setToolTip(self._t("folders.compute_mismatch"))
        self.btn_cancel.setEnabled(is_building)

    def _build_folder_manager_payload(self) -> dict:
        return {
            "folders": [
                {
                    "path": str(self.folder_list.item(row).data(Qt.ItemDataRole.UserRole)),
                    "checked": bool(self.folder_list.item(row).data(FOLDER_CHECKED_ROLE)),
                    "label": str(self.folder_list.item(row).data(FOLDER_LABEL_ROLE) or ""),
                }
                for row in range(self.folder_list.count())
            ],
            "base_folder": str(self._base_folder.path) if self._base_folder is not None else None,
        }

    def _restore_persisted_state(self) -> None:
        self._restore_folder_manager_state()
        self._restore_build_settings()
        self._restore_error_view_settings()

    def _restore_folder_manager_state(self) -> None:
        payload = self._settings_service.load_folder_manager_payload()
        if not payload:
            return
        self.folder_list.blockSignals(True)
        try:
            for folder_entry in payload.get("folders", []):
                path = folder_entry.get("path")
                if not path:
                    continue
                folder_path = Path(path)
                if not folder_path.exists():
                    continue
                item = self._append_folder_item(folder_path, checked=bool(folder_entry.get("checked", False)))
                item.setData(FOLDER_LABEL_ROLE, str(folder_entry.get("label") or folder_path.name))
            base_folder = payload.get("base_folder")
            if base_folder:
                base_path = Path(base_folder)
                if base_path.exists():
                    self._base_folder = FolderSpec(path=base_path, label=base_path.name)
        finally:
            self.folder_list.blockSignals(False)

    def _build_build_settings_payload(self) -> dict:
        return {
            "comparison_mode": str(getattr(self.mode_combo.currentData(), "value", self.mode_combo.currentData())),
            "thumbnail_size": int(self.thumbnail_size_spin.value()),
            "layout_mode": str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE),
            "total_frames": int(self.total_frames_spin.value()),
            "frames_per_row": int(self.frames_per_row_spin.value()),
            "rows": int(self.matrix_rows_spin.value()),
            "columns": int(self.matrix_columns_spin.value()),
        }

    def _restore_build_settings(self) -> None:
        payload = self._settings_service.load_build_settings_payload()
        if not payload:
            return
        blockers = [
            QSignalBlocker(self.mode_combo),
            QSignalBlocker(self.thumbnail_size_spin),
            QSignalBlocker(self.layout_mode_combo),
            QSignalBlocker(self.total_frames_spin),
            QSignalBlocker(self.frames_per_row_spin),
            QSignalBlocker(self.matrix_rows_spin),
            QSignalBlocker(self.matrix_columns_spin),
        ]
        _ = blockers
        comparison_mode_value = str(payload.get("comparison_mode") or DEFAULT_COMPARISON_MODE.value)
        comparison_mode = ComparisonMode(comparison_mode_value) if comparison_mode_value in ComparisonMode._value2member_map_ else DEFAULT_COMPARISON_MODE
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(comparison_mode))
        self.thumbnail_size_spin.setValue(int(payload.get("thumbnail_size", self.thumbnail_size_spin.value())))
        layout_mode = str(payload.get("layout_mode") or DEFAULT_MATRIX_LAYOUT_MODE)
        layout_mode_index = self.layout_mode_combo.findData(layout_mode)
        self.layout_mode_combo.setCurrentIndex(layout_mode_index if layout_mode_index >= 0 else 0)
        self.total_frames_spin.setValue(int(payload.get("total_frames", DEFAULT_TOTAL_FRAMES)))
        self.frames_per_row_spin.setValue(int(payload.get("frames_per_row", DEFAULT_FRAMES_PER_ROW)))
        self.matrix_rows_spin.setValue(int(payload.get("rows", DEFAULT_MATRIX_ROWS)))
        self.matrix_columns_spin.setValue(int(payload.get("columns", DEFAULT_MATRIX_COLUMNS)))

    def _build_error_view_payload(self) -> dict:
        return {
            "gradient_name": self.gradient_selector.selected_gradient(),
            "error_window": [float(value) for value in self.gradient_range_selector.error_window()],
            "score_view_mode": str(self.error_score_view_combo.currentData() or DEFAULT_SCORE_VIEW_MODE),
        }

    def _restore_error_view_settings(self) -> None:
        payload = self._settings_service.load_error_view_payload()
        if not payload:
            return
        blockers = [QSignalBlocker(self.error_score_view_combo)]
        _ = blockers
        score_view_mode = str(payload.get("score_view_mode") or DEFAULT_SCORE_VIEW_MODE)
        self.error_score_view_combo.setCurrentIndex(self.error_score_view_combo.findData(score_view_mode))
        self.gradient_selector.set_selected_gradient(str(payload.get("gradient_name") or DEFAULT_GRADIENT_NAME), emit_signal=False)
        self.gradient_range_selector.set_gradient_name(self.gradient_selector.selected_gradient())
        error_window = payload.get("error_window") or list(DEFAULT_ERROR_WINDOW)
        if isinstance(error_window, (list, tuple)) and len(error_window) == 2:
            self.gradient_range_selector.set_error_window(float(error_window[0]), float(error_window[1]))

    def _persist_state(self) -> None:
        self._settings_service.save_folder_manager_payload(self._build_folder_manager_payload())
        self._settings_service.save_build_settings_payload(self._build_build_settings_payload())
        self._settings_service.save_error_view_payload(self._build_error_view_payload())
        self._settings_service.sync()

    def shutdown(self) -> None:
        if self._worker is not None:
            request_cancel = getattr(self._worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        self._persist_state()




