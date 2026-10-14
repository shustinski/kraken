"""main_window: setup ui."""
from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QWidget,
)


class SetupUiMixin:
    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.setMinimumHeight(64)

        self.drag_handle = QLabel('...', self)
        self.drag_handle.setFixedWidth(18)
        self.drag_handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drag_handle.setToolTip(str(self._texts.get('queue_drag_tip', 'Drag to reorder')))
        layout.addWidget(self.drag_handle)

        self.mode_label = QLabel(self)
        self.mode_label.setFixedWidth(170)
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.mode_label.setStyleSheet('padding-right: 10px;')
        layout.addWidget(self.mode_label)

        self.name_edit = QLineEdit(self)
        self.name_edit.setMinimumHeight(30)
        self.name_edit.setStyleSheet('padding-left: 8px; padding-right: 8px;')
        self.name_edit.textChanged.connect(lambda _text: self._fit_name_edit_to_text())
        self.name_edit.editingFinished.connect(self._emit_name_changed)
        layout.addWidget(self.name_edit, 1)

        self.status_label = QLabel(self)
        self.status_label.setMinimumWidth(105)
        layout.addWidget(self.status_label)

        self.progress_widget = QWidget(self)
        progress_layout = QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(3)
        self.epoch_progress_bar = QProgressBar(self.progress_widget)
        self.batch_progress_bar = QProgressBar(self.progress_widget)
        self.validation_progress_bar = QProgressBar(self.progress_widget)
        for progress_bar in (
            self.epoch_progress_bar,
            self.batch_progress_bar,
            self.validation_progress_bar,
        ):
            progress_bar.setRange(0, 100)
            progress_bar.setMinimumWidth(145)
            progress_bar.setMaximumWidth(190)
            progress_bar.setFixedHeight(23)
            progress_layout.addWidget(progress_bar, 1)
        # Compatibility alias for recognition and older callers.
        self.progress_bar = self.epoch_progress_bar
        layout.addWidget(self.progress_widget)

        self.completion_label = QLabel(self)
        self.completion_label.setFixedWidth(24)
        self.completion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.completion_label)

        self.pause_continue_button = self._make_button('>', self.pause_continue_requested)
        self.up_button = self._make_button('^', self.move_up_requested)
        self.down_button = self._make_button('v', self.move_down_requested)
        self.remove_button = self._make_button('X', self.remove_requested)
        self.retry_button = self._make_button('R', self.retry_requested)
        self.pause_continue_button.setText('>')
        self.up_button.setText('▲')
        self.down_button.setText('▼')
        self.remove_button.setText('×')
        self.retry_button.setText('↻')
        self._style_action_button(self.pause_continue_button)
        self._style_action_button(self.up_button)
        self._style_action_button(self.down_button)
        self._style_action_button(self.retry_button)
        self._style_action_button(self.remove_button, foreground='#ffb4b4', border='#7f2a2a')
        for button in (
            self.pause_continue_button,
            self.retry_button,
            self.up_button,
            self.down_button,
            self.remove_button,
        ):
            layout.addWidget(button)
