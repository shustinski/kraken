"""main_window: apply ui language."""
from __future__ import annotations
from neuralimage.lib.ui_texts import get_ui_section


class ApplyUiLanguageMixin:
    def apply_ui_language(self, language: str) -> None:
        self._ui_language = "en" if language == "en" else "ru"
        t = get_ui_section("main_window")
        self._texts = t
        if self._sample_count_pending:
            self.set_samples_count_loading()
        else:
            self.set_samples_count(self._sample_count_value)
        self.work_mode_group.setTitle(t["mode"])
        self.rb_train_and_recognition.setText(t["mode_train_and_rec"])
        self.rb_further_train_model.setText(t["mode_ft_and_rec"])
        self.rb_recognition.setText(t["mode_rec"])
        self.rb_train_only.setText(t["mode_train"])
        self.rb_continue_training.setText(t.get("mode_continue_training", "Continue training"))
        self.source_title_label.setText(t["source"])
        self.recursive_file_search_check_box.setText(str(t.get("recursive_file_search", "Search in subfolders")))
        self.recursive_file_search_check_box.setToolTip(
            str(t.get("recursive_file_search_tip", "Find source files recursively under the selected folder."))
        )
        self.result_title_label.setText(t["result"])
        self.sample_path_group.setTitle(t["sample"])
        self.sample_path.setToolTip(t["sample_tip"])
        self.label_path.setToolTip(t["label_tip"])
        if self.sample_src_title_label is not None:
            self.sample_src_title_label.setText(t["sample_src"])
        if self.label_path_title_label is not None:
            self.label_path_title_label.setText(t["labels"])
        self.model_title_label.setText(t["model"])
        self.epochs_title_label.setText(t["epochs"])
        self.btn_start.setText(t["start"])
        self.btn_stop.setText(t["stop"])
        self.queue_group.setTitle(t["queue"])
        self.queue_dock.setWindowTitle(t["queue"])
        self.simple_workflows_group.setTitle(t.get("simple_workflows_group", "Simple workflows"))
        self.btn_simple_conductors.setText(t.get("simple_workflow_conductors", "Conductor recognition"))
        self.btn_simple_contacts.setText(t.get("simple_workflow_contacts", "Contact recognition"))
        self.btn_simple_memory.setText(t.get("simple_workflow_memory", "Memory recognition"))
        self._update_simple_workflow_label()
        self.progress_group.setTitle(t["progress_group"])
        if self.progress_epochs_title_label is not None:
            self.progress_epochs_title_label.setText(t["progress_epochs"])
        if self.progress_batches_title_label is not None:
            self.progress_batches_title_label.setText(t["progress_batches"])
        if self.progress_recognition_title_label is not None:
            self.progress_recognition_title_label.setText(t["progress_recognition"])
        self._update_recognition_speed_label()
        self.preview_group.setTitle(t["preview_group"])
        self._set_preview_frame_name(self._current_preview_sample_name)
        self._set_preview_mode(self._current_preview_mode)
        self.statusBar().clearMessage()
        if self._settings_menu is not None:
            self._settings_menu.setTitle(t["menu_settings"])
        if self._file_menu is not None:
            self._file_menu.setTitle(t.get("menu_file", "Файл"))
        if self._view_menu is not None:
            self._view_menu.setTitle(t["menu_view"])
        if self._info_menu is not None:
            self._info_menu.setTitle(t["menu_help"])
        if self._plugins_menu is not None:
            self._plugins_menu.setTitle(t.get("menu_plugins", "Plugins"))
        if self._update_channel_menu is not None:
            self._update_channel_menu.setTitle(t.get("menu_update_channel", "Update channel"))
        if self._update_menu is not None:
            self._update_menu.setTitle(t.get("menu_update", "Update"))
        if self._open_config_action is not None:
            self._open_config_action.setText(t.get("menu_open_config", "Открыть"))
        if self._language_menu is not None:
            self._language_menu.setTitle(t.get("menu_language", "Язык"))
        if self._theme_menu is not None:
            self._theme_menu.setTitle(t.get("menu_theme", "Тема"))
        if self._ui_mode_menu is not None:
            self._ui_mode_menu.setTitle(t.get("menu_ui_mode", "Interface mode"))
        if self._settings_sample_action is not None:
            self._settings_sample_action.setText(t.get("menu_sample", "Выборка"))
        if self._settings_train_action is not None:
            self._settings_train_action.setText(t.get("menu_train", "Обучение"))
        if self._settings_pred_action is not None:
            self._settings_pred_action.setText(t.get("menu_pred", "Распознавание"))
        if self._metrics_toggle_action is not None:
            self._metrics_toggle_action.setText(t.get("menu_metrics", "Панель графиков"))
        if self._log_toggle_action is not None:
            self._log_toggle_action.setText(t.get("menu_log_panel", "Панель лога"))
        if self._settings_toggle_action is not None:
            self._settings_toggle_action.setText(t.get("menu_settings_panel", "Панель настроек"))
        if hasattr(self, "batch_preview_action"):
            self.batch_preview_action.setText(t.get("menu_batch_preview", "Превью пакета обучения"))
        if hasattr(self, "release_memory_action"):
            self.release_memory_action.setText(t.get("menu_release_memory", "Освободить память GPU"))
        if hasattr(self, "open_tic_tac_toe_action"):
            self.open_tic_tac_toe_action.setText(
                t.get("menu_open_tic_tac_toe", "Крестики-нолики (нейросеть)")
            )
        if self._open_validation_gradient_action is not None:
            self._open_validation_gradient_action.setText(
                t.get("menu_open_validation_gradient", "Open Validation gradient")
            )
        if self._help_action is not None:
            self._help_action.setText(t.get("menu_open_help", "Открыть справку"))
        if self._changelog_action is not None:
            self._changelog_action.setText(t.get("menu_open_changelog", "Список изменений"))
        if self._check_updates_action is not None:
            self._check_updates_action.setText(t.get("menu_check_updates", "Проверить обновления"))
        for channel, action in self._update_channel_actions.items():
            action.setText(self._format_update_channel_label(channel))
        if hasattr(self, "ui_language_ru_action"):
            self.ui_language_ru_action.setText(t.get("lang_ru", "Русский"))
        if hasattr(self, "ui_language_en_action"):
            self.ui_language_en_action.setText(t.get("lang_en", "English"))
        if hasattr(self, "theme_dark_action"):
            self.theme_dark_action.setText(t.get("theme_dark", "Темная"))
        if hasattr(self, "theme_light_action"):
            self.theme_light_action.setText(t.get("theme_light", "Светлая"))
        if self._ui_mode_simple_action is not None:
            self._ui_mode_simple_action.setText(t.get("menu_ui_mode_simple", "Simple"))
        if self._ui_mode_advanced_action is not None:
            self._ui_mode_advanced_action.setText(t.get("menu_ui_mode_advanced", "Advanced"))
        self.log_dock.setWindowTitle(t.get("log_dock_title", "Лог"))
        if self.settings_dock is not None:
            self.settings_dock.setWindowTitle(t.get("settings_dock_title", "Настройки"))
        self._update_memory_runtime_label()
        if self._last_validation_metrics is not None:
            iou, dice, f1 = self._last_validation_metrics
            self.validation_quality_label.setText(self._format_validation_quality_text(iou, dice, f1))
        else:
            self.validation_quality_label.setText(t["validation_quality_default"])
        if self._last_performance_metrics is not None:
            self.performance_label.setText(
                self._format_performance_text(**self._last_performance_metrics)
            )
        else:
            self.performance_label.setText(t["performance_label_default"])
        self._sync_language_menu_checks()
