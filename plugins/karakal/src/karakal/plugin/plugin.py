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
        if host is not None and callable(getattr(host, "publish_quality", None)):
            self._widget.qualityPublicationRequested.connect(host.publish_quality)
            self._widget.set_kraken_publish_available(True)
        return self._widget

    def shutdown(self) -> None:
        if self._widget is not None:
            self._widget.shutdown()
            self._widget = None
        self._host = None
