"""Minimal plugin interfaces used to embed Cartograph into a host application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QWidget


@runtime_checkable
class PluginHost(Protocol):
    def settings(self) -> QSettings | None: ...

    def logger(self) -> Any: ...

    def task_runner(self) -> Any: ...

    def open_path(self, path: Path) -> None: ...


@runtime_checkable
class WidgetPlugin(Protocol):
    plugin_id: str
    display_name: str

    def create_widget(self, host: PluginHost | None = None, parent: QWidget | None = None) -> QWidget: ...

    def shutdown(self) -> None: ...
