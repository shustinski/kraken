"""main_presenter: connect view signals."""
from __future__ import annotations
from neuralimage.lib.message_bus import MessageBus
import neuralimage.presenter.state_sync as state_sync


class ConnectViewSignalsMixin:
    # ------------------------------------------------------------------ #
    #   Подписки
    # ------------------------------------------------------------------ #
    def _conncet_to_message_bus(self):
        self.message_bus = MessageBus()
        self.message_bus.subscribe('logging', self._log_message_emit)
        self.message_bus.subscribe('training', self._train_message_emit)
        self.message_bus.subscribe('metrics', self._metrics_message_emit)
        self.message_bus.subscribe('warning', self._warning_message_emit)
        self.message_bus.subscribe('error', self._error_message_emit)

    def _connect_view_signals(self):
        v = self.view
        v.sample_type_changed.connect(self._on_sample_type_changed)
        v.le_epochs.valueChanged.connect(
            lambda value: self.settings_panel.epochs_spinbox.setValue(int(value))
        )

        v.source_path_requested.connect(self._choose_source_folder)
        v.result_path_requested.connect(self._choose_result_folder)

        v.jpg_path_requested.connect(self._choose_jpg_label_path)
        v.label_path_requested.connect(self._chosse_cif_label_path)

        v.model_path_requested.connect(self._choose_model_path)
        v.open_config_requested.connect(self._on_open_config_requested)

        v.start_requested.connect(self._on_start_requested)
        v.stop_requested.connect(self._on_stop_requested)
        v.queue_context_remove_requested.connect(self._on_queue_context_remove_requested)
        v.queue_properties_requested.connect(self._on_queue_properties_requested)
        v.queue_report_requested.connect(self._on_queue_report_requested)
        v.queue_retry_requested.connect(self._on_queue_retry_requested)
        v.queue_pause_continue_requested.connect(self._on_queue_pause_continue_requested)
        v.queue_move_up_requested.connect(self._on_queue_move_up_requested)
        v.queue_move_down_requested.connect(self._on_queue_move_down_requested)
        v.queue_rows_reordered.connect(self._on_queue_rows_reordered)
        v.queue_task_name_changed.connect(self._on_queue_task_name_changed)
        v.batch_preview_visibility_changed.connect(self._on_batch_preview_visibility_changed)
        v.release_memory_requested.connect(self._on_release_memory_requested)
        v.open_validation_gradient_requested.connect(self._on_open_validation_gradient_requested)
        v.update_check_requested.connect(self._on_update_check_requested)
        v.update_channel_selected.connect(self._on_update_channel_selected)
        v.ui_language_selected.connect(self._on_ui_language_selected)
        v.theme_selected.connect(self._on_theme_selected)
        v.ui_mode_selected.connect(self._on_ui_mode_selected)
        v.simple_workflow_requested.connect(self._on_simple_workflow_requested)
        if hasattr(v, 'recursive_file_search_changed'):
            v.recursive_file_search_changed.connect(self._on_recursive_file_search_changed)

        v.request_close.connect(self._on_close_requested)

    def _connect_settings_signal(self):
        v = self.settings_panel
        v.epochs_spinbox.valueChanged.connect(lambda value: self.view.le_epochs.setValue(int(value)))
        v.cut_slider_shifted.connect(self._calculate_expected_samples)
        v.horisontal_rotate_clicked.connect(self._calculate_expected_samples)
        v.vertical_rotate_clicked.connect(self._calculate_expected_samples)
        v.model_changed.connect(self._calculate_expected_samples)
        v.sample_size_changed.connect(self._set_max_shift)
        v.optimizer_settings_changed.connect(self._update_settings_window_state)
        v.validation_settings_changed.connect(self._update_settings_window_state)
        v.validation_image_path_requested.connect(self._choose_validation_image_folder)
        v.validation_label_path_requested.connect(self._choose_validation_label_folder)
        v.reset_defaults_requested.connect(self._reset_settings_to_defaults)
        v.augmentation_preview_requested.connect(self._open_augmentation_preview)
        v.ui_language_changed.connect(self.view.apply_ui_language)
        v.rare_patch_editor_requested.connect(self._open_rare_patch_editor)
        if hasattr(v, 'configuration_validity_changed'):
            v.configuration_validity_changed.connect(self._on_settings_configuration_validity_changed)

    def _on_settings_configuration_validity_changed(self, valid: bool, _messages: object) -> None:
        self._settings_configuration_valid = bool(valid)
        self._validate_start_button()

    def _on_recursive_file_search_changed(self) -> None:
        self._update_settings_window_state()
        self._calculate_expected_samples()

    def _publish_log_message(self, message: str) -> None:
        state_dict = object.__getattribute__(self, '__dict__')
        bus = state_dict.get('message_bus')
        if bus is None:
            return
        try:
            bus.publish('logging', str(message))
        except Exception:
            return

    # ------------------------------------------------------------------ #
    #   Инициализация UI
    # ------------------------------------------------------------------ #
    def _load_initial_state(self):
        state_sync.load_initial_state(self)

    def _load_main_window_settings(self):
        state_sync.load_main_window_settings(self)

    def _load_settings_panel_settings(self):
        state_sync.load_settings_panel_settings(self)

    def _set_initial_sample_count_state(self) -> None:
        state_sync.set_initial_sample_count_state(self)

    def _current_view_main_window_mode_entry(self) -> dict[str, object]:
        return state_sync.current_view_main_window_mode_entry(self)

    def _store_view_state_for_mode(self, mode: str) -> None:
        state_sync.store_view_state_for_mode(self, mode)

    def _restore_mode_state_to_view(self, mode: str) -> None:
        state_sync.restore_mode_state_to_view(self, mode)

    def _update_main_window_state(self):
        state_sync.update_main_window_state(self)

    def _restore_work_mode_ui(self):
        state_sync.restore_work_mode_ui(self)
