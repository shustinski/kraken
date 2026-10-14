"""main_window: style action button."""
from __future__ import annotations
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QToolButton,
)


class StyleActionButtonMixin:
    def _make_button(self, text: str, signal: pyqtSignal) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setFixedSize(30, 30)
        button.clicked.connect(lambda _checked=False, sig=signal: sig.emit(self._row))
        return button

    @staticmethod
    def _style_action_button(
        button: QToolButton,
        *,
        foreground: str = '#edf3fb',
        border: str = '#30445a',
    ) -> None:
        button.setStyleSheet(
            (
                'QToolButton {'
                'background-color: #1d2733; '
                f'border: 1px solid {border}; '
                'border-radius: 4px; '
                f'color: {foreground}; '
                'font-weight: 700; '
                'font-size: 17px; '
                'padding: 0px;'
                '}'
                'QToolButton:hover { background-color: #283342; border-color: #46627f; }'
                'QToolButton:disabled { background-color: #151c25; color: #6f7c89; border-color: #243142; }'
            )
        )

    def _emit_name_changed(self) -> None:
        self.name_changed.emit(self._row, self.name_edit.text())
