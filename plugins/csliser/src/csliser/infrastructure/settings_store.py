from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QSettings


@dataclass(frozen=True, slots=True)
class WindowSettings:
    destination: str = ""
    frame_expression: str = ""
    frames_per_row: int = 135
    font_size: int = 14
    add_extension_prefix: bool = True
    selection_mode: str = "rectangle"
    operation: str = "copy"


def build_csliser_settings() -> QSettings:
    root = os.getenv("VIALANET_SETTINGS_DIR") or os.getenv("CSLISER_SETTINGS_DIR")
    if root:
        settings_root = Path(root)
        settings_root.mkdir(parents=True, exist_ok=True)
        return QSettings(str(settings_root / "ViaLaNet_CSliser.ini"), QSettings.Format.IniFormat)
    return QSettings("ViaLaNet", "CSliser")


class WindowSettingsStore:
    def __init__(self, settings_factory: Callable[[], QSettings] | None = None) -> None:
        self._settings_factory = settings_factory or build_csliser_settings

    def load(self) -> WindowSettings:
        settings = self._settings_factory()
        result = WindowSettings(
            destination=settings.value("paths/destination", "", type=str),
            frame_expression=settings.value("frames/expression", "", type=str),
            frames_per_row=settings.value("frames/per_row", 135, type=int),
            font_size=settings.value("appearance/font_size", 14, type=int),
            add_extension_prefix=settings.value("output/add_extension_prefix", True, type=bool),
            selection_mode=settings.value("frames/selection_mode", "rectangle", type=str),
            operation=settings.value("operation/current", "copy", type=str),
        )
        settings.sync()
        return result

    def save(self, payload: WindowSettings) -> None:
        settings = self._settings_factory()
        settings.setValue("paths/destination", payload.destination)
        settings.setValue("frames/expression", payload.frame_expression)
        settings.setValue("frames/per_row", payload.frames_per_row)
        settings.setValue("appearance/font_size", payload.font_size)
        settings.setValue("output/add_extension_prefix", payload.add_extension_prefix)
        settings.setValue("frames/selection_mode", payload.selection_mode)
        settings.setValue("operation/current", payload.operation)
        settings.sync()
