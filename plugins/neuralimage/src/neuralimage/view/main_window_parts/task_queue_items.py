"""main_window: task queue items."""
from __future__ import annotations
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QListWidgetItem,
)
from neuralimage.lib.shared_styles import resolve_shared_style_path
from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.view.tic_tac_toe_dialog import TicTacToeDialog

from neuralimage.view.main_window_parts.responsive_preview_label import (
    _save_persisted_ui_mode,
    load_qss_from_resource,
)


class TaskQueueItemsMixin:
    def _switch_start_stop(self, show_stop: bool):
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)
        if show_stop:
            self._reset_runtime_metrics()
            self.metrics_panel.clear()
            self._batch_points_by_epoch.clear()
            self._last_epoch_progress = (0, 0)
            self._last_batch_progress = (0, 0)
            self._last_recognition_progress = (0, 0)
            self._show_progress_mode("idle")
            self._set_progress_bar(self.epoch_progress_bar, 0, 0)
            self._set_progress_bar(self.batch_progress_bar, 0, 0)
            self._set_progress_bar(self.recognition_progress_bar, 0, 0)
            self._update_recognition_speed_label()
            self.preview_image_label.clear()
            self.preview_label_label.clear()
            self.preview_output_label.clear()
            self._set_preview_mode("train")
            self._set_preview_frame_name("")
            self._update_memory_runtime_label()
            self.validation_quality_label.setText(get_ui_section("main_window")["validation_quality_default"])
            self.performance_label.setText(get_ui_section("main_window")["performance_label_default"])

    def _open_tic_tac_toe_dialog(self):
        if self._tic_tac_toe_dialog is None:
            self._tic_tac_toe_dialog = TicTacToeDialog(self)
        self._tic_tac_toe_dialog.show()
        self._tic_tac_toe_dialog.raise_()
        self._tic_tac_toe_dialog.activateWindow()

    def get_selected_queue_row(self) -> int:
        return self.queue_list.currentRow()

    def set_task_queue_items(self, items: list, selected_row: int = -1):
        previous_scroll_value = self.queue_list.verticalScrollBar().value()
        previous_selected_row = self.queue_list.currentRow()
        texts = self._main_texts()
        normalized_items: list[dict] = []
        for item in items:
            if isinstance(item, dict):
                normalized_items.append(dict(item))
            else:
                normalized_items.append(
                    {
                        'work_mode': str(item),
                        'work_mode_label': str(item),
                        'display_name': '',
                        'status': 'waiting',
                        'status_label': '',
                    }
                )
        self._queue_items = normalized_items
        self.queue_list.clear()
        for row, item in enumerate(normalized_items):
            list_item = QListWidgetItem()
            list_item.setSizeHint(QSize(920, 66))
            status = str(item.get('status', 'waiting'))
            flags = list_item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            if status not in {'in_progress', 'finished_success', 'finished_error'}:
                flags |= Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
            list_item.setFlags(flags)
            widget = TaskQueueRowWidget(row, item, texts, self.queue_list)
            widget.name_changed.connect(self.queue_task_name_changed.emit)
            widget.pause_continue_requested.connect(self.queue_pause_continue_requested.emit)
            widget.move_up_requested.connect(self.queue_move_up_requested.emit)
            widget.move_down_requested.connect(self.queue_move_down_requested.emit)
            widget.remove_requested.connect(self.queue_context_remove_requested.emit)
            widget.retry_requested.connect(self.queue_retry_requested.emit)
            self.queue_list.addItem(list_item)
            self.queue_list.setItemWidget(list_item, widget)
        resolved_selected_row = selected_row if selected_row >= 0 else previous_selected_row
        if 0 <= resolved_selected_row < len(normalized_items):
            self.queue_list.setCurrentRow(resolved_selected_row)
        self.queue_list.verticalScrollBar().setValue(
            max(0, min(previous_scroll_value, self.queue_list.verticalScrollBar().maximum()))
        )

    def update_task_queue_item_progress(
        self,
        task_id: int,
        current: int,
        total: int,
        progress_kind: str = '',
    ) -> None:
        for row, item in enumerate(self._queue_items):
            if int(item.get('task_id', -1)) != int(task_id):
                continue
            item['progress_current'] = int(current)
            item['progress_total'] = int(total)
            item['progress_kind'] = str(progress_kind or '')
            list_item = self.queue_list.item(row)
            widget = self.queue_list.itemWidget(list_item) if list_item is not None else None
            if isinstance(widget, TaskQueueRowWidget):
                widget.update_item(row, item, self._main_texts())
            return

    def set_stylesheet(self, style):
        self.setStyleSheet(style)

    def _theme_qss_path(self, theme: str) -> str:
        if theme == "light":
            return str(resolve_shared_style_path('style.qss'))
        return str(resolve_shared_style_path('dark_modern.qss'))

    def _sync_language_menu_checks(self) -> None:
        if hasattr(self, "ui_language_ru_action"):
            self.ui_language_ru_action.setChecked(self._ui_language == "ru")
        if hasattr(self, "ui_language_en_action"):
            self.ui_language_en_action.setChecked(self._ui_language == "en")

    def _sync_theme_menu_checks(self) -> None:
        if hasattr(self, "theme_dark_action"):
            self.theme_dark_action.setChecked(self._theme == "dark")
        if hasattr(self, "theme_light_action"):
            self.theme_light_action.setChecked(self._theme == "light")

    def _sync_ui_mode_menu_checks(self) -> None:
        if self._ui_mode_simple_action is not None:
            self._ui_mode_simple_action.setChecked(self._ui_mode == "simple")
        if self._ui_mode_advanced_action is not None:
            self._ui_mode_advanced_action.setChecked(self._ui_mode == "advanced")

    def _apply_theme(self, theme: str) -> None:
        self._theme = "light" if theme == "light" else "dark"
        qss_path = self._theme_qss_path(self._theme)
        self.set_stylesheet(load_qss_from_resource(qss_path))
        self._sync_theme_menu_checks()

    def apply_theme(self, theme: str) -> None:
        self._apply_theme(theme)

    def _handle_theme_action(self, theme: str) -> None:
        self._apply_theme(theme)
        self.theme_selected.emit(self._theme)

    def _handle_ui_mode_action(self, mode: str) -> None:
        self.apply_ui_mode(mode)
        self.ui_mode_selected.emit(self._ui_mode)

    def apply_ui_mode(self, mode: str) -> None:
        normalized_mode = "simple" if mode == "simple" else "advanced"
        self._ui_mode = normalized_mode
        is_simple = normalized_mode == "simple"
        self.simple_workflows_group.setVisible(is_simple)
        self.log_dock.setVisible(not is_simple)
        self.metrics_panel.setVisible(not is_simple)
        if self.settings_dock is not None:
            self.settings_dock.setVisible(not is_simple)
        if self._metrics_toggle_action is not None:
            self._metrics_toggle_action.setVisible(not is_simple)
        if self._log_toggle_action is not None:
            self._log_toggle_action.setVisible(not is_simple)
        if self._settings_toggle_action is not None:
            self._settings_toggle_action.setVisible(not is_simple)
        if self._settings_menu is not None:
            self._settings_menu.menuAction().setVisible(not is_simple)
        self.apply_work_mode(self._current_work_mode)
        self._sync_ui_mode_menu_checks()
        _save_persisted_ui_mode(self._ui_mode)

    def current_ui_mode(self) -> str:
        return self._ui_mode

    def _handle_simple_workflow_click(self, profile_key: str) -> None:
        self.set_simple_workflow_profile(profile_key)
        self.simple_workflow_requested.emit(profile_key)

    def set_simple_workflow_profile(self, profile_key: str | None) -> None:
        self._selected_simple_workflow = str(profile_key) if profile_key else None
        profile_buttons = {
            'conductors': self.btn_simple_conductors,
            'contacts': self.btn_simple_contacts,
            'memory': self.btn_simple_memory,
        }
        for key, button in profile_buttons.items():
            button.setChecked(key == self._selected_simple_workflow)
        self._update_simple_workflow_label()

    def _simple_workflow_display_name(self, profile_key: str | None) -> str:
        mapping = {
            'conductors': self.btn_simple_conductors.text(),
            'contacts': self.btn_simple_contacts.text(),
            'memory': self.btn_simple_memory.text(),
        }
        return mapping.get(str(profile_key), self._main_texts().get('simple_workflow_none', 'Not selected'))

    def _update_simple_workflow_label(self) -> None:
        t = self._main_texts()
        template = str(t.get('simple_workflow_selected_template', 'Current profile: {profile}'))
        self.simple_workflow_label.setText(
            template.format(profile=self._simple_workflow_display_name(self._selected_simple_workflow))
        )
