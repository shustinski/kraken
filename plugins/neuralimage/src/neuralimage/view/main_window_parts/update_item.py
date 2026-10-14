"""main_window: update item."""
from __future__ import annotations


class UpdateItemMixin:
    def update_item(self, row: int, item: dict, texts: dict[str, str]) -> None:
        self._row = int(row)
        self._texts = texts
        status = str(item.get('status', 'waiting'))
        self.mode_label.setText(str(item.get('work_mode_label', item.get('work_mode', ''))))
        self.name_edit.blockSignals(True)
        self.name_edit.setText(str(item.get('display_name', '')))
        self.name_edit.blockSignals(False)
        self._fit_name_edit_to_text()
        self.status_label.setText(str(item.get('status_label', status)))

        progress_kind = str(item.get('progress_kind', '') or '')
        is_recognition_progress = progress_kind == 'recognition_progress'
        if is_recognition_progress:
            self._set_row_progress(
                self.epoch_progress_bar,
                str(self._texts.get('progress_recognition', 'Recognition')),
                int(item.get('progress_current', 0) or 0),
                int(item.get('progress_total', 0) or 0),
            )
        else:
            self._set_row_progress(
                self.epoch_progress_bar,
                str(self._texts.get('progress_epochs', 'Epochs')),
                int(item.get('epoch_progress_current', 0) or 0),
                int(item.get('epoch_progress_total', 0) or 0),
            )
            self._set_row_progress(
                self.batch_progress_bar,
                str(self._texts.get('progress_batches', 'Batches')),
                int(item.get('batch_progress_current', 0) or 0),
                int(item.get('batch_progress_total', 0) or 0),
            )
            self._set_row_progress(
                self.validation_progress_bar,
                str(self._texts.get('progress_validation', 'Validation')),
                int(item.get('validation_progress_current', 0) or 0),
                int(item.get('validation_progress_total', 0) or 0),
            )
        self.progress_widget.setVisible(status == 'in_progress')
        self.batch_progress_bar.setVisible(status == 'in_progress' and not is_recognition_progress)
        self.validation_progress_bar.setVisible(
            status == 'in_progress'
            and not is_recognition_progress
        )

        self.completion_label.setText('')
        self.completion_label.setStyleSheet('')
        if status == 'finished_success':
            self.completion_label.setText('OK')
            self.completion_label.setStyleSheet('color: #2e7d32; font-weight: 700;')
        elif status == 'finished_error':
            self.completion_label.setText('X')
            self.completion_label.setStyleSheet('color: #d32f2f; font-weight: 700;')

        is_running = status == 'in_progress'
        is_pausing = status == 'pausing'
        is_paused = status == 'paused'
        is_finished = status in {'stopped', 'finished_success', 'finished_error'}
        self.drag_handle.setEnabled(not is_running and not is_finished)
        self.drag_handle.setVisible(not is_running and not is_finished)
        self.pause_continue_button.setVisible(is_running or is_pausing or is_paused)
        self.pause_continue_button.setEnabled(not is_pausing)
        self.pause_continue_button.setText('▮▮' if is_running or is_pausing else '▶')
        self.pause_continue_button.setToolTip(
            str(texts.get('queue_pause_task', 'Pause')) if is_running else str(texts.get('queue_continue_task', 'Continue'))
        )
        self.up_button.setVisible(True)
        self.down_button.setVisible(True)
        self.remove_button.setVisible(True)
        self.up_button.setEnabled(not is_running and not is_finished)
        self.down_button.setEnabled(not is_running and not is_finished)
        self.remove_button.setEnabled(True)
        self.retry_button.setVisible(is_finished)
        self.up_button.setToolTip(str(texts.get('queue_move_up', 'Move up')))
        self.down_button.setToolTip(str(texts.get('queue_move_down', 'Move down')))
        self.remove_button.setToolTip(
            str(texts.get('stop', 'Stop')) if is_running else str(texts.get('queue_remove', 'Remove from queue'))
        )
        self.retry_button.setToolTip(str(texts.get('queue_retry', 'Retry')))
