"""main_window: apply work mode."""
from __future__ import annotations
from neuralimage.lib.data_interfaces import WorkMode


class ApplyWorkModeMixin:
    def set_batch_preview_enabled(self, enabled: bool) -> None:
        action = getattr(self, "batch_preview_action", None)
        if action is not None:
            action.setChecked(enabled)
        preview_group = getattr(self, "preview_group", None)
        if preview_group is not None:
            preview_group.setVisible(bool(enabled))

    def is_batch_preview_enabled(self) -> bool:
        action = getattr(self, "batch_preview_action", None)
        if action is None:
            return True
        return bool(action.isChecked())

    def _current_side_panel_width(self) -> int:
        if self.settings_dock is None or not self.settings_dock.isVisible():
            return 0
        return self.settings_dock.width()

    def _compute_target_side_panel_width(self) -> int:
        if self.settings_dock is None:
            return 0
        return self.settings_dock.width()

    def _show_side_panel_fully(self):
        if self.settings_dock is None:
            return
        self.settings_dock.show()
        self.settings_dock.raise_()

    def show_settings_dock(self) -> None:
        self._show_side_panel_fully()

    def _hide_side_panel(self):
        if self.settings_dock is None:
            return
        self.settings_dock.hide()

    def _toogle_button_clicked(self):
        if self.settings_dock is None:
            return
        panel_visible = self.settings_dock.isVisible()
        if panel_visible:
            self._hide_side_panel()
        else:
            self._show_side_panel_fully()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_window_layout_save()

    def apply_work_mode(self, mode: str) -> None:
        self._current_work_mode = str(mode or '').strip()
        resolved_mode = self._current_work_mode
        training_only = resolved_mode in {WorkMode.train_only.value, WorkMode.continue_training.value}
        recognition_only = resolved_mode == WorkMode.recognition_only.value
        uses_model = resolved_mode in {
            WorkMode.recognition_only.value,
            WorkMode.further_training.value,
            WorkMode.continue_training.value,
        }

        if not resolved_mode:
            self.lbl_source.setEnabled(True)
            self.recursive_file_search_check_box.setEnabled(True)
            self.lbl_result.setEnabled(True)
            self.sample_path_group.setEnabled(True)
            self.sample_path_group.setVisible(True)
            self.model_path.setEnabled(True)
        else:
            self.lbl_source.setEnabled(not training_only)
            self.recursive_file_search_check_box.setEnabled(True)
            self.lbl_result.setEnabled(not training_only)
            self.sample_path_group.setEnabled(not recognition_only)
            self.sample_path_group.setVisible(not recognition_only)
            self.model_path.setEnabled(uses_model)

        show_model = True if not resolved_mode else uses_model
        show_epochs = False
        self.model_title_label.setVisible(show_model)
        self.model_path.setVisible(show_model)
        self.epochs_title_label.setVisible(show_epochs)
        self.le_epochs.setVisible(show_epochs)
