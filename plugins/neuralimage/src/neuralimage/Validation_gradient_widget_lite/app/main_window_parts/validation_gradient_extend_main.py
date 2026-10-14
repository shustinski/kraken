"""main_window: validation gradient extend main."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QMainWindow,
)
from ...ui.ui_constants import (
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
)

class ValidationGradientExtendMainWindow(QMainWindow):
    """Standalone host window for the extended widget."""

    def __init__(self) -> None:
        super().__init__()
        self._widget = ValidationGradientExtendWidget(self)
        self.setWindowTitle(self._widget._t("window.title"))
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setCentralWidget(self._widget)

    def plugin_widget(self) -> ValidationGradientExtendWidget:
        return self._widget

    def closeEvent(self, event) -> None:
        self._widget.shutdown()
        super().closeEvent(event)
