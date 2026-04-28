"""Expose Karakal through a small plugin entrypoint."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from .api import PluginHost

from ..app.main_window import KarakalWidget


class KarakalPlugin:
    """Expose the Karakal widget through a host-facing plugin object."""

    plugin_id = "karakal"
    display_name = "Karakal"

    def __init__(self) -> None:
        self._widget: KarakalWidget | None = None
        self._host: PluginHost | None = None

    def create_widget(self, host: PluginHost | None = None, parent: QWidget | None = None) -> KarakalWidget:
        self._host = host
        self._widget = KarakalWidget(parent)
        return self._widget

    def shutdown(self) -> None:
        if self._widget is not None:
            self._widget.shutdown()
            self._widget = None
        self._host = None
