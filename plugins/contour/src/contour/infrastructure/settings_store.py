from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSettings

from ..application.dto import PersistedPaths
from ..application.processing import DisplaySettings

VIA_PRESETS_SETTINGS_KEY = "via_search/user_presets"


def _build_contour_settings() -> QSettings:
    root = os.getenv("VIALANET_SETTINGS_DIR") or os.getenv("NEURALIMAGE_SETTINGS_DIR")
    if root:
        settings_root = Path(root)
        settings_root.mkdir(parents=True, exist_ok=True)
        return QSettings(
            str(settings_root / "ViaLaNet_Contour.ini"),
            QSettings.Format.IniFormat,
        )
    return QSettings("ViaLaNet", "Contour")


class WidgetPathSettingsStore:
    def __init__(self, settings_factory: Callable[[], QSettings] | None = None) -> None:
        self._settings_factory = settings_factory or _build_contour_settings

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
        self._settings_factory = settings_factory or _build_contour_settings

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
            "autosave_on_frame_transition": settings.value("display/autosave_on_frame_transition", False, type=bool),
            "vector_geom_clip_on_sync": settings.value("display/vector_geom_clip_on_sync", True, type=bool),
            "vector_geom_min_outer_area": settings.value("display/vector_geom_min_outer_area", 9.0, type=float),
            "vector_geom_min_hole_area": settings.value("display/vector_geom_min_hole_area", 0.0, type=float),
            "vector_geom_merge_on_edit": settings.value("display/vector_geom_merge_on_edit", True, type=bool),
            "vector_geom_spike_angle_deg": settings.value("display/vector_geom_spike_angle_deg", 30.0, type=float),
            "vector_geom_drop_triangles": settings.value("display/vector_geom_drop_triangles", True, type=bool),
            "main_splitter_sizes": settings.value("display/main_splitter_sizes", []),
        }
        settings.sync()
        return payload

    def save(self, payload: dict[str, object]) -> None:
        settings = self._settings_factory()
        for key, value in payload.items():
            settings.setValue(f"display/{key}", value)
        settings.sync()


class WidgetViaPresetSettingsStore:
    def __init__(self, settings_factory: Callable[[], QSettings] | None = None) -> None:
        self._settings_factory = settings_factory or _build_contour_settings

    def load(self) -> dict[str, dict[str, object]]:
        settings = self._settings_factory()
        raw_payload = settings.value(VIA_PRESETS_SETTINGS_KEY, "{}", type=str)
        settings.sync()
        try:
            payload = json.loads(str(raw_payload or "{}"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(name): dict(value) for name, value in payload.items() if isinstance(value, dict)}

    def save(self, presets: dict[str, dict[str, object]]) -> None:
        settings = self._settings_factory()
        settings.setValue(VIA_PRESETS_SETTINGS_KEY, json.dumps(presets, ensure_ascii=False, sort_keys=True))
        settings.sync()
