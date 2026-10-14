"""main_presenter: shutdown."""
from __future__ import annotations
import gc
from dataclasses import replace
from PyQt6 import QtCore
from neuralimage.application.dto import (
    MainWindowState,
    SettingsState,
    normalize_main_window_mode_state,
)
from neuralimage.application.services import (
    QueuedTask,
    build_workflow_parameters,
)
from neuralimage.application.services.training_artifacts import build_training_artifact_dir
from neuralimage.infrastructure.config.state_store import (
    WORKFLOW_SNAPSHOT_FILENAME,
    load_workflow_snapshot,
    save_workflow_snapshot,
)
from neuralimage.lib.data_interfaces import (
    normalize_work_mode,
)
from neuralimage.lib.update_checker import (
    load_selected_update_channel,
    load_update_client_config,
    save_selected_update_channel,
)
from neuralimage.view.task_properties_dialog import TaskPropertiesDialog
import neuralimage.presenter.plugin_flow as plugin_flow
import neuralimage.presenter.state_sync as state_sync
import neuralimage.presenter.task_flow as task_flow
import neuralimage.presenter.update_flow as update_flow

from neuralimage.presenter.main_presenter_parts.general_neural_handler_thread import GeneralNeuralHandlerThread


class ShutdownMixin:
    def _on_sample_type_changed(self, typ: str):
        """typ = ''train_and_recognition'', 'recognition_only', further_training."""
        mode = normalize_work_mode(typ)
        previous_mode = normalize_work_mode(getattr(self.main_window_state, 'work_mode', ''))
        if previous_mode:
            self._store_view_state_for_mode(previous_mode)
        self.main_window_state.work_mode = mode
        self._restore_mode_state_to_view(mode)
        self.view.apply_work_mode(mode)
        if hasattr(self.settings_panel, 'sync_business_logic_controls'):
            self.settings_panel.sync_business_logic_controls(mode)
        self._calculate_expected_samples()
        self._validate_start_button()

    def _on_start_requested(self):
        task_flow.on_start_requested(self)

    def _on_stop_requested(self):
        task_flow.on_stop_requested(self)

    def _on_release_memory_requested(self):
        gc.collect()
        self.message_bus.publish('logging', 'Выполнена очистка памяти Python.')

    def _on_ui_language_selected(self, language: str):
        self.settings_panel.set_ui_language(language)
        QtCore.QTimer.singleShot(0, lambda: self._refresh_queue_view(selected_row=self.view.get_selected_queue_row()))

    def _on_theme_selected(self, theme: str):
        self.view.apply_theme(theme)

    def _on_ui_mode_selected(self, mode: str):
        normalized_mode = 'advanced' if mode == 'advanced' else 'simple'
        self.view.apply_ui_mode(normalized_mode)
        self.main_window_state.ui_mode = normalized_mode
        self._save_main_window_to_qsettings()

    def _refresh_update_client_config(self) -> None:
        update_flow.refresh_update_client_config(
            self,
            load_update_client_config_fn=load_update_client_config,
            load_selected_update_channel_fn=load_selected_update_channel,
        )

    def _on_update_channel_selected(self, channel: str) -> None:
        update_flow.on_update_channel_selected(
            self,
            channel,
            save_selected_update_channel_fn=save_selected_update_channel,
            load_update_client_config_fn=load_update_client_config,
            load_selected_update_channel_fn=load_selected_update_channel,
        )

    def _on_simple_workflow_requested(self, preset_name: str):
        preset_path = self.SIMPLE_WORKFLOW_PRESETS.get(str(preset_name))
        if preset_path is None:
            self.view.show_warning.emit('Неизвестный простой профиль.')
            return
        try:
            main_state, settings_state = load_workflow_snapshot(preset_path)
        except (OSError, ValueError) as error:
            self.view.show_warning.emit(f'Не удалось загрузить профиль: {error}')
            return

        restored_main_state = replace(
            main_state,
            work_mode=self.main_window_state.work_mode,
            source_folder=self.main_window_state.source_folder,
            result_folder=self.main_window_state.result_folder,
            label_folder=self.main_window_state.label_folder,
            sample_folder=self.main_window_state.sample_folder,
            model_path=self.main_window_state.model_path,
            epochs=self.main_window_state.epochs,
            ui_mode='simple',
            mode_state=normalize_main_window_mode_state(getattr(self.main_window_state, 'mode_state', {})),
        )
        self._restore_task_state_to_ui(
            restored_main_state,
            settings_state,
            log_message=f'Загружен простой профиль: {preset_path.name}.',
        )
        self.view.set_simple_workflow_profile(str(preset_name))
        self._save_windows_to_qsettings()

    def _on_update_check_requested(self) -> None:
        self._start_update_check(manual=True)

    def _on_queue_context_remove_requested(self, row: int):
        self._remove_queue_row(row)

    def _remove_queue_row(self, row: int):
        task_flow.remove_queue_row(self, row)

    def _on_queue_properties_requested(self, row: int):
        task_flow.on_queue_properties_requested(self, row, dialog_cls=TaskPropertiesDialog)

    def _on_queue_report_requested(self, row: int):
        task_flow.on_queue_report_requested(self, row)

    def _on_queue_retry_requested(self, row: int):
        task_flow.on_queue_retry_requested(self, row)

    def _on_queue_pause_continue_requested(self, row: int):
        task_flow.on_queue_pause_continue_requested(self, row)

    def _on_queue_move_up_requested(self, row: int):
        task_flow.on_queue_move_requested(self, row, -1)

    def _on_queue_move_down_requested(self, row: int):
        task_flow.on_queue_move_requested(self, row, 1)

    def _on_queue_rows_reordered(self, source_row: int, target_row: int):
        task_flow.on_queue_rows_reordered(self, source_row, target_row)

    def _on_queue_task_name_changed(self, row: int, display_name: str):
        task_flow.on_queue_task_name_changed(self, row, display_name)

    def _on_task_restore_requested(self, main_state: MainWindowState, settings_state: SettingsState):
        task_flow.on_task_restore_requested(self, main_state, settings_state)

    def _restore_task_state_to_ui(
        self,
        main_state: MainWindowState,
        settings_state: SettingsState,
        *,
        log_message: str = 'Параметры задачи восстановлены в интерфейсе.',
    ):
        state_sync.restore_task_state_to_ui(
            self,
            main_state,
            settings_state,
            log_message=log_message,
        )

    def _on_queue_pause_toggle_requested(self):
        task_flow.on_queue_pause_toggle_requested(self)

    def _refresh_queue_view(self, selected_row: int = -1, selected_task_id: int | None = None):
        task_flow.refresh_queue_view(
            self,
            selected_row=selected_row,
            selected_task_id=selected_task_id,
        )

    def _start_next_task_if_possible(self):
        if self._shutdown_requested:
            return
        task_flow.start_next_task_if_possible(self)

    def _start_task(self, task: QueuedTask):
        task_flow.start_task(
            self,
            task,
            handler_thread_cls=GeneralNeuralHandlerThread,
            workflow_builder=build_workflow_parameters,
            artifact_dir_builder=build_training_artifact_dir,
            workflow_snapshot_saver=save_workflow_snapshot,
            workflow_snapshot_filename=WORKFLOW_SNAPSHOT_FILENAME,
        )

    def _on_task_finished(self):
        task_flow.on_task_finished(self)

    def _clear_validation_gradient_window_refs(self) -> None:
        plugin_flow.clear_validation_gradient_window_refs(self)

    def _on_open_validation_gradient_requested(self) -> None:
        plugin_flow.open_validation_gradient_requested(
            self,
            window_cls=plugin_flow.ValidationGradientPluginWindow,
        )

    def _shutdown_validation_gradient_plugin(self) -> None:
        plugin_flow.shutdown_validation_gradient_plugin(self)

    def shutdown(self, *, wait_ms: int = 15000) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_requested = True
        self._shutdown_complete = True

        self._sample_count_debounce_timer.stop()
        self._sample_count_cancel_event.set()
        self._debounced_sample_count_request = None
        self._pending_sample_count_request = None
        self._latest_sample_count_request_id = self._sample_count_request_serial + 1

        handler = self.neuaral_handler
        if handler is not None:
            self._processing_session.request_stop()
            self._stop_worker_object(handler, 'active neural task')
            self._wait_for_qthread(handler, wait_ms, 'active neural task')
            if not self._is_qthread_running(handler):
                self.neuaral_handler = None

        self._stop_owned_qthread('_remote_connection_check', wait_ms, 'remote connection check')
        self._stop_owned_qthread('_rare_patch_editor_prepare_thread', wait_ms, 'rare patch preparation')
        self._stop_owned_qthread('_update_check_thread', min(wait_ms, 3000), 'update check')
        self._stop_owned_qthread('_update_download_thread', min(wait_ms, 3000), 'update download')

        sample_thread = self._sample_count_worker_thread
        if sample_thread is not None and sample_thread.is_alive():
            sample_thread.join(timeout=1.0)
        if sample_thread is None or not sample_thread.is_alive():
            self._sample_count_worker_thread = None

        self._shutdown_validation_gradient_plugin()
