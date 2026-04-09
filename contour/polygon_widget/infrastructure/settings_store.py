from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSettings

from ..application.dto import PersistedPaths


def _build_polygon_widget_settings() -> QSettings:
    root = os.getenv("VIALANET_SETTINGS_DIR") or os.getenv("NEURALIMAGE_SETTINGS_DIR")
    if root:
        settings_root = Path(root)
        settings_root.mkdir(parents=True, exist_ok=True)
        return QSettings(
            str(settings_root / "ViaLaNet_PolygonWidget.ini"),
            QSettings.Format.IniFormat,
        )
    return QSettings("ViaLaNet", "PolygonWidget")


class WidgetPathSettingsStore:
    def __init__(self, settings_factory: Callable[[], QSettings] | None = None) -> None:
        self._settings_factory = settings_factory or _build_polygon_widget_settings

    def load(self) -> PersistedPaths:
        settings = self._settings_factory()
        paths = PersistedPaths(
            input_directory=settings.value("paths/input_directory", "", type=str),
            cif_directory=settings.value("paths/cif_directory", "", type=str),
            output_directory=settings.value("paths/output_directory", "", type=str),
        )
        settings.sync()
        return paths

    def save(self, paths: PersistedPaths) -> None:
        settings = self._settings_factory()
        settings.setValue("paths/input_directory", paths.input_directory)
        settings.setValue("paths/cif_directory", paths.cif_directory)
        settings.setValue("paths/output_directory", paths.output_directory)
        settings.sync()
