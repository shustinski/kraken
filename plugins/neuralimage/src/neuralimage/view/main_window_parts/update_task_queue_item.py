"""main_window: update task queue item."""
from __future__ import annotations
from neuralimage.application.dto import MainWindowState


class UpdateTaskQueueItemMixin:
    def _handle_ui_language_action(self, language: str) -> None:
        if self.settings_dock is not None and hasattr(self.settings_dock, "set_ui_language"):
            self.settings_dock.set_ui_language(language)
        else:
            self.apply_ui_language(language)
        self.ui_language_selected.emit(language)

    def set_source_path(self, path: str):
        self.lbl_source.setText(path)

    def set_recursive_file_search(self, enabled: bool) -> None:
        self.recursive_file_search_check_box.setChecked(bool(enabled))

    def is_recursive_file_search_enabled(self) -> bool:
        return self.recursive_file_search_check_box.isChecked()

    def set_result_path(self, path: str):
        self.lbl_result.setText(path)

    def set_samples_count(self, total_samples: int) -> None:
        try:
            self._sample_count_value = int(total_samples)
        except (TypeError, ValueError):
            self._sample_count_value = 0
        self._sample_count_pending = False
        template = str(
            self._texts.get("samples_count_template", self._texts.get("samples_count", "Кадров в выборке: {count}"))
        )
        if "{count}" in template:
            text = template.format(count=self._sample_count_value)
        else:
            text = template
        self.sample_count_top_label.setText(text)

    def set_samples_count_loading(self) -> None:
        self._sample_count_pending = True
        self.sample_count_top_label.setText(str(self._texts.get("samples_count_loading", "Идет расчет...")))

    def set_label_path(self, path: str):
        self.label_path.setText(path)

    def set_jpg_path(self, path: str):
        self.sample_path.setText(path)

    def restore_from_dataclass(self, state: MainWindowState):
        self.set_source_path(state.source_folder)
        self.set_result_path(state.result_folder)
        self.label_path.setText(state.label_folder)
        self.sample_path.setText(state.sample_folder)
        self.model_path.setText(state.model_path)
        self.le_epochs.setValue(state.epochs)

    def closeEvent(self, event):
        if self._close_allowed:
            self._window_layout_save_timer.stop()
            self._save_window_layout()
            event.accept()
            return
        self.request_close.emit()
        event.ignore()

    def update_task_queue_item_training_progress(
        self,
        task_id: int,
        *,
        epoch_current: int | None = None,
        epoch_total: int | None = None,
        batch_current: int | None = None,
        batch_total: int | None = None,
        validation_current: int | None = None,
        validation_total: int | None = None,
    ) -> None:
        for row, item in enumerate(self._queue_items):
            if int(item.get('task_id', -1)) != int(task_id):
                continue
            if epoch_current is not None:
                item['epoch_progress_current'] = max(0, int(epoch_current))
            if epoch_total is not None:
                item['epoch_progress_total'] = max(0, int(epoch_total))
            if batch_current is not None:
                item['batch_progress_current'] = max(0, int(batch_current))
            if batch_total is not None:
                item['batch_progress_total'] = max(0, int(batch_total))
            if validation_current is not None:
                item['validation_progress_current'] = max(0, int(validation_current))
            if validation_total is not None:
                item['validation_progress_total'] = max(0, int(validation_total))
            if validation_current is not None:
                item['progress_kind'] = 'validation_progress'
            elif batch_current is not None:
                item['progress_kind'] = 'train_batch_progress'
            else:
                item['progress_kind'] = 'train_epoch_progress'
            list_item = self.queue_list.item(row)
            widget = self.queue_list.itemWidget(list_item) if list_item is not None else None
            if isinstance(widget, TaskQueueRowWidget):
                widget.update_item(row, item, self._main_texts())
            return

    def allow_close(self):
        self._close_allowed = True
        self.close()
