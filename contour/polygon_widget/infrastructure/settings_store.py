from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSettings

from ..application.dto import PersistedPaths
from ..application.processing import DisplaySettings


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
            dataset_directory=settings.value("paths/dataset_directory", "", type=str),
        )
        settings.sync()
        return paths

    def save(self, paths: PersistedPaths) -> None:
        settings = self._settings_factory()
        settings.setValue("paths/input_directory", paths.input_directory)
        settings.setValue("paths/cif_directory", paths.cif_directory)
        settings.setValue("paths/output_directory", paths.output_directory)
        settings.setValue("paths/dataset_directory", paths.dataset_directory)
        settings.sync()


class WidgetDisplaySettingsStore:
    def __init__(self, settings_factory: Callable[[], QSettings] | None = None) -> None:
        self._settings_factory = settings_factory or _build_polygon_widget_settings

    def load(self) -> dict[str, object]:
        settings = self._settings_factory()
        defaults = DisplaySettings()
        payload: dict[str, object] = {
            "external_color": settings.value("display/external_color", defaults.external_color, type=str),
            "hole_color": settings.value("display/hole_color", defaults.hole_color, type=str),
            "selected_color": settings.value("display/selected_color", defaults.selected_color, type=str),
            "vertex_color": settings.value("display/vertex_color", defaults.vertex_color, type=str),
            "line_width": settings.value("display/line_width", defaults.line_width, type=float),
            "vertex_size": settings.value("display/vertex_size", defaults.vertex_size, type=float),
            "fill_opacity": settings.value("display/fill_opacity", defaults.fill_opacity, type=float),
            "show_vertices": settings.value("display/show_vertices", defaults.show_vertices, type=bool),
            "show_labels": settings.value("display/show_labels", defaults.show_labels, type=bool),
            "show_neighbor_frames": settings.value("display/show_neighbor_frames", False, type=bool),
            "neighbor_columns": settings.value("display/neighbor_columns", 3, type=int),
            "neighbor_max_grid": settings.value("display/neighbor_max_grid", 7, type=int),
            "neighbor_opacity": settings.value("display/neighbor_opacity", 0.35, type=float),
            "neighbor_overlap_pixels": settings.value("display/neighbor_overlap_pixels", 0, type=int),
        }
        settings.sync()
        return payload

    def save(self, payload: dict[str, object]) -> None:
        settings = self._settings_factory()
        for key, value in payload.items():
            settings.setValue(f"display/{key}", value)
        settings.sync()
