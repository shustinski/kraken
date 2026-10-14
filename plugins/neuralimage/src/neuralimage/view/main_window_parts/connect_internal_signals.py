"""main_window: connect internal signals."""
from __future__ import annotations
from neuralimage.lib.data_interfaces import WorkMode

from neuralimage.view.main_window_parts.responsive_preview_label import TaskQueueListWidget


class ConnectInternalSignalsMixin:
    def connect_internal_signals(self):
        self.rb_train_and_recognition.clicked.connect(
            lambda _: self.sample_type_changed.emit(WorkMode.train_and_recognition.value)
        )
        self.rb_recognition.clicked.connect(lambda _: self.sample_type_changed.emit(WorkMode.recognition_only.value))
        self.rb_further_train_model.clicked.connect(
            lambda _: self.sample_type_changed.emit(WorkMode.further_training.value)
        )
        self.rb_train_only.clicked.connect(lambda _: self.sample_type_changed.emit(WorkMode.train_only.value))
        self.rb_continue_training.clicked.connect(
            lambda _: self.sample_type_changed.emit(WorkMode.continue_training.value)
        )

        self.lbl_source.clicked.connect(lambda: self.source_path_requested.emit())
        self.recursive_file_search_check_box.toggled.connect(lambda _: self.recursive_file_search_changed.emit())
        self.lbl_result.clicked.connect(lambda: self.result_path_requested.emit())

        self.label_path.clicked.connect(lambda: self.label_path_requested.emit())
        self.sample_path.clicked.connect(lambda: self.jpg_path_requested.emit())

        self.model_path.clicked.connect(lambda: self.model_path_requested.emit())
        if self._open_config_action is not None:
            self._open_config_action.triggered.connect(lambda _checked=False: self.open_config_requested.emit())
        if self._settings_sample_action is not None and self.settings_dock is not None:
            self._settings_sample_action.triggered.connect(lambda _checked=False: self._show_settings_page("base"))
        if self._settings_train_action is not None and self.settings_dock is not None:
            self._settings_train_action.triggered.connect(
                lambda _checked=False: self._show_settings_page("training")
            )
        if self._settings_pred_action is not None and self.settings_dock is not None:
            self._settings_pred_action.triggered.connect(
                lambda _checked=False: self._show_settings_page("recognition")
            )

        self.le_epochs.valueChanged.connect(lambda _: self.epochs_changed.emit())

        self.btn_start.clicked.connect(lambda: self.start_requested.emit())
        self.btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self.queue_list.customContextMenuRequested.connect(self._show_queue_context_menu)
        if isinstance(self.queue_list, TaskQueueListWidget):
            self.queue_list.rows_reordered.connect(self.queue_rows_reordered.emit)
        if hasattr(self, "batch_preview_action"):
            self.batch_preview_action.toggled.connect(self.batch_preview_visibility_changed.emit)
            self.batch_preview_action.toggled.connect(self.set_batch_preview_enabled)
        if hasattr(self, "release_memory_action"):
            self.release_memory_action.triggered.connect(self.release_memory_requested.emit)
        if hasattr(self, "open_tic_tac_toe_action"):
            self.open_tic_tac_toe_action.triggered.connect(self.open_tic_tac_toe_requested.emit)
        if self._open_validation_gradient_action is not None:
            self._open_validation_gradient_action.triggered.connect(self.open_validation_gradient_requested.emit)
        if self._check_updates_action is not None:
            self._check_updates_action.triggered.connect(self.update_check_requested.emit)
        if hasattr(self, "ui_language_ru_action"):
            self.ui_language_ru_action.triggered.connect(lambda: self._handle_ui_language_action("ru"))
        if hasattr(self, "ui_language_en_action"):
            self.ui_language_en_action.triggered.connect(lambda: self._handle_ui_language_action("en"))
        if hasattr(self, "theme_dark_action"):
            self.theme_dark_action.triggered.connect(lambda: self._handle_theme_action("dark"))
        if hasattr(self, "theme_light_action"):
            self.theme_light_action.triggered.connect(lambda: self._handle_theme_action("light"))
        if self._ui_mode_simple_action is not None:
            self._ui_mode_simple_action.triggered.connect(lambda: self._handle_ui_mode_action("simple"))
        if self._ui_mode_advanced_action is not None:
            self._ui_mode_advanced_action.triggered.connect(lambda: self._handle_ui_mode_action("advanced"))
        self.btn_simple_conductors.clicked.connect(lambda: self._handle_simple_workflow_click("conductors"))
        self.btn_simple_contacts.clicked.connect(lambda: self._handle_simple_workflow_click("contacts"))
        self.btn_simple_memory.clicked.connect(lambda: self._handle_simple_workflow_click("memory"))
        self.open_tic_tac_toe_requested.connect(self._open_tic_tac_toe_dialog)

        self.log_message.connect(self._append_log)
        self.log_message_with_delete_last.connect(self.append_with_delete_previous)
        self.metrics_message.connect(self._append_metrics)
        self.enable_start.connect(self._set_start_enabled)
        self.show_info.connect(self._show_info_message)
        self.show_warning.connect(self._show_warning_message)
        self.toggle_start_stop.connect(self._switch_start_stop)
