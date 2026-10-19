"""main_presenter: open rare patch editor."""
from __future__ import annotations
from pathlib import Path
import json
from dataclasses import replace
from PyQt6 import QtCore
from neuralimage.application.dto import (
    SettingsState,
)
from neuralimage.infrastructure.config.state_store import (
    load_workflow_snapshot,
)
from neuralimage.lib.file_func import filter_files
from neuralimage.lib.logging_policy import should_forward_log_event
import neuralimage.presenter.state_sync as state_sync
import neuralimage.presenter.task_flow as task_flow
from neuralimage.presenter.background_workers import (
    RarePatchEditorPreparationThread as PresenterRarePatchEditorPreparationThread,
)

from neuralimage.presenter.main_presenter_parts.rare_patch_editor_preparation import _tk_filedialog


class OpenRarePatchEditorMixin:
    def _open_rare_patch_editor(self):
        self._update_main_window_state()
        sample_folder = Path(self.main_window_state.sample_folder)
        label_folder = Path(self.main_window_state.label_folder)
        if not sample_folder.is_dir() or not label_folder.is_dir():
            self.view.show_warning.emit(
                'Выберите существующие папки с изображениями и метками перед открытием редактора редких областей.'
            )
            return

        if self._rare_patch_editor_prepare_thread is not None:
            self.message_bus.publish(
                'logging',
                'Подготовка меток для редактора редких областей уже выполняется.',
            )
            return

        if filter_files(label_folder, ('.cif',)):
            self._start_rare_patch_editor_cif_conversion(sample_folder, label_folder)
            return

        self._open_rare_patch_editor_dialog(sample_folder, label_folder)

    def _start_rare_patch_editor_cif_conversion(self, sample_folder: Path, label_folder: Path) -> None:
        self.message_bus.publish(
            'logging',
            'Обнаружены CIF-метки. Готовлю binary_cif для редактора редких областей.',
        )
        self._set_rare_patch_editor_preparing(True)
        thread = PresenterRarePatchEditorPreparationThread(
            sample_folder=sample_folder,
            label_folder=label_folder,
            message_bus=self.message_bus,
        )
        thread.prepared.connect(self._on_rare_patch_editor_prepared)
        thread.finished.connect(thread.deleteLater)
        self._rare_patch_editor_prepare_thread = thread
        thread.start()

    def _set_rare_patch_editor_preparing(self, active: bool) -> None:
        button = getattr(self.settings_panel, 'edit_rare_regions_button', None)
        if button is not None:
            button.setEnabled(not active)

    @QtCore.pyqtSlot(str, str, str)
    def _on_rare_patch_editor_prepared(
        self,
        sample_folder: str,
        resolved_label_folder: str,
        error_message: str,
    ) -> None:
        self._set_rare_patch_editor_preparing(False)
        self._rare_patch_editor_prepare_thread = None

        if error_message:
            self.view.show_warning.emit(error_message)
            return

        self.view.set_label_path(resolved_label_folder)
        self.main_window_state.label_folder = resolved_label_folder
        self._open_rare_patch_editor_dialog(Path(sample_folder), Path(resolved_label_folder))

    def _open_rare_patch_editor_dialog(self, sample_folder: Path, label_folder: Path) -> None:
        from neuralimage.lib.rare_patch_masks import collect_matching_sample_label_pairs
        from neuralimage.view.rare_patch_editor_dialog import RarePatchEditorDialog

        _pairs, error_message = collect_matching_sample_label_pairs(sample_folder, label_folder)
        if error_message is not None:
            self.view.show_warning.emit(error_message)
            return

        dialog = RarePatchEditorDialog(sample_folder, label_folder, self.view)
        dialog.exec()

    def _log_message_emit(self, data):
        if not should_forward_log_event('logging', data):
            return
        self.view.log_message.emit(data)

    def _train_message_emit(self, data):
        if not should_forward_log_event('training', data):
            return
        self.view.log_message_with_delete_last.emit(data)

    def _metrics_message_emit(self, data):
        self._queue_metrics_received.emit(data)
        self.view.metrics_message.emit(data)

    def _warning_message_emit(self, data):
        message = str(data) if data is not None else 'Предупреждение.'
        self.view.show_warning.emit(message)
        self.view.log_message.emit(f'Предупреждение: {message}')

    def _error_message_emit(self, data):
        message = str(data) if data is not None else 'Произошла ошибка выполнения.'
        self._queue_error_received.emit(message)
        self.view.show_warning.emit(message)
        self.view.log_message.emit(f'Ошибка: {message}')

    def _on_queue_metrics_received(self, data) -> None:
        task_flow.on_metrics_message(self, data)

    def _on_queue_error_received(self, message: str) -> None:
        task_flow.on_error_message(self, message)

    def _update_cut_mode(self) -> str:
        return state_sync.update_cut_mode(self)

    # ------------------------------------------------------------------ #
    #   Сохранение окон в QSettings
    # ------------------------------------------------------------------ #
    def _save_windows_to_qsettings(self):
        self._update_main_window_state()
        self._update_settings_window_state()

        self._save_main_window_to_qsettings()
        self._save_nn_settings_to_qsettings()

    def _save_main_window_to_qsettings(self):
        self._state_store.save_main_window_state(self.main_window_state)

    def _save_nn_settings_to_qsettings(self):
        state = getattr(self, "settings_state", None) or SettingsState()
        self.settings_state = state
        self._state_store.save_settings_state(state)

    def _reset_settings_to_defaults(self):
        state_sync.reset_settings_to_defaults(self)

    def _choose_source_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_source_path(path)
            self.main_window_state.source_folder = path
            self._validate_start_button()

    def _choose_jpg_label_path(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_jpg_path(path)
            self.main_window_state.sample_folder = path
            self.sample_calculator.set_path(Path(path))
            self._calculate_expected_samples()
            self._validate_start_button()

    def _chosse_cif_label_path(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_label_path(path)
            self.main_window_state.label_folder = path

            self._validate_start_button()

    def _choose_validation_image_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.settings_panel.set_validation_image_path(path)
            self.settings_state.validation_image_folder = path
            self._validate_start_button()

    def _choose_validation_label_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.settings_panel.set_validation_label_path(path)
            self.settings_state.validation_label_folder = path
            self._validate_start_button()

    def _choose_result_folder(self):
        path = _tk_filedialog('folder')
        if path:
            self.view.set_result_path(path)
            self.main_window_state.result_folder = path
            self._validate_start_button()

    def _choose_model_path(self):
        path = _tk_filedialog('file', [("Модель", ".pth")])
        if path:
            self.view.model_path.setText(path)
            self._validate_start_button()

    def _on_open_config_requested(self):
        path = _tk_filedialog('file', [('JSON', '.json')])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("format") == "neuralimage-inference-settings":
                from neuralimage.application.services.inference_settings import restore_inference_settings
                from neuralimage.lib.data_interfaces import WorkMode

                settings_state = restore_inference_settings(payload, self.settings_state)
                main_state = replace(self.main_window_state, work_mode=WorkMode.recognition_only.value)
            else:
                main_state, settings_state = load_workflow_snapshot(path)
        except (OSError, ValueError) as error:
            self.view.show_warning.emit(f'Не удалось загрузить параметры из файла: {error}')
            return

        self._restore_task_state_to_ui(
            main_state,
            settings_state,
            log_message='Параметры восстановлены из файла.',
        )
        self._save_windows_to_qsettings()
