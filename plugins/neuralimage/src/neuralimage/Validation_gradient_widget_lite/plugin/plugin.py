"""Expose the lite mismatch-only widget through a small plugin entrypoint."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from .api import PluginHost

from ..app.main_window import ValidationGradientLiteWidget


class ValidationGradientLitePlugin:
    """Expose the lite widget through a host-facing plugin object."""

    plugin_id = 'validation_gradient_widget_lite'
    display_name = 'Validation Gradient Widget Lite'

    def __init__(self) -> None:
        self._widget: ValidationGradientLiteWidget | None = None
        self._host: PluginHost | None = None

    def create_widget(self, host: PluginHost | None = None, parent: QWidget | None = None) -> ValidationGradientLiteWidget:
        self._host = host
        self._widget = ValidationGradientLiteWidget(parent)
        return self._widget

    def shutdown(self) -> None:
        if self._widget is not None:
            self._widget.shutdown()
            self._widget = None
        self._host = None

