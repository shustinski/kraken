"""Define the minimal plugin interfaces used to embed the validation widget into a host application."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QWidget


@runtime_checkable
class PluginHost(Protocol):
    """Describe the host services available to the validation widget plugin."""

    def settings(self) -> QSettings | None:
        """Return host-owned settings storage when it exists."""
        ...

    def logger(self) -> Any:
        """Return the host logger used for diagnostics."""
        ...

    def task_runner(self) -> Any:
        """Return the host task runner used for long-running work."""
        ...

    def open_path(self, path: Path) -> None:
        """Ask the host to open a filesystem path for the user."""
        ...


@runtime_checkable
class WidgetPlugin(Protocol):
    """Describe the plugin interface expected by the host application."""

    plugin_id: str
    display_name: str

    def create_widget(self, host: PluginHost | None = None, parent: QWidget | None = None) -> QWidget:
        """Create and return the plugin widget instance."""
        ...

    def shutdown(self) -> None:
        """Persist plugin state and stop running background work."""
        ...
