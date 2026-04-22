from __future__ import annotations

from typing import Callable

from PyQt6 import QtCore, QtWidgets


class ValidationGradientPluginWindow(QtWidgets.QMainWindow):
    def __init__(self, plugin, widget, title: str, on_closed: Callable[[], None], parent=None):
        super().__init__(parent)
        self._plugin = plugin
        self._on_closed = on_closed
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(str(title))
        self.setCentralWidget(widget)
        self.resize(1400, 900)

    def closeEvent(self, event) -> None:
        try:
            if self._plugin is not None:
                self._plugin.shutdown()
        finally:
            self._plugin = None
            try:
                self._on_closed()
            finally:
                super().closeEvent(event)


def clear_validation_gradient_window_refs(presenter) -> None:
    presenter._validation_gradient_window = None
    presenter._validation_gradient_plugin = None


def open_validation_gradient_requested(presenter, *, window_cls=ValidationGradientPluginWindow) -> None:
    window = presenter._validation_gradient_window
    if window is not None:
        window.show()
        window.raise_()
        window.activateWindow()
        return

    from Validation_gradient_widget_lite import ValidationGradientLitePlugin

    plugin = ValidationGradientLitePlugin()
    try:
        widget = plugin.create_widget(parent=None)
    except Exception as exc:
        presenter.view.show_warning.emit(f'Failed to open Validation Gradient Lite: {exc}')
        return

    title = getattr(plugin, 'display_name', 'Validation Gradient Widget Lite')
    window = window_cls(
        plugin=plugin,
        widget=widget,
        title=str(title),
        on_closed=lambda: clear_validation_gradient_window_refs(presenter),
        parent=presenter.view,
    )
    presenter._validation_gradient_plugin = plugin
    presenter._validation_gradient_window = window
    window.show()
    window.raise_()
    window.activateWindow()


def shutdown_validation_gradient_plugin(presenter) -> None:
    window = presenter._validation_gradient_window
    plugin = presenter._validation_gradient_plugin
    if window is not None:
        presenter._validation_gradient_window = None
        presenter._validation_gradient_plugin = None
        window.close()
        return
    if plugin is not None:
        try:
            plugin.shutdown()
        finally:
            presenter._validation_gradient_plugin = None
