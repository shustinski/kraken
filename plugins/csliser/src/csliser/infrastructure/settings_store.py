from __future__ import annotations

import json
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
    dynamic_extensions: bool = False


@dataclass(frozen=True, slots=True)
class CSliserPreset:
    sources: tuple[str, ...] = ()
    destination: str = ""
    frames: str = ""
    row_frames: str = "135"

    @classmethod
    def from_legacy_dict(cls, payload: dict[str, object]) -> CSliserPreset:
        sources = payload.get("source", ())
        if not isinstance(sources, list):
            sources = ()
        return cls(
            sources=tuple(str(item) for item in sources),
            destination=str(payload.get("destination", "")),
            frames=str(payload.get("frames", "")),
            row_frames=str(payload.get("row_frames", "135")),
        )

    def to_legacy_dict(self, *, include_sources: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "frames": self.frames,
            "row_frames": self.row_frames,
        }
        if include_sources:
            result["source"] = list(self.sources)
            result["destination"] = self.destination
        return result


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
            dynamic_extensions=settings.value("source/dynamic_extensions", False, type=bool),
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
        settings.setValue("source/dynamic_extensions", payload.dynamic_extensions)
        settings.sync()

    def load_presets(self, *, include_sources: bool) -> dict[str, CSliserPreset]:
        settings = self._settings_factory()
        raw = settings.value(_preset_key(include_sources), "{}", type=str)
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(name): CSliserPreset.from_legacy_dict(value)
            for name, value in payload.items()
            if isinstance(value, dict)
        }

    def save_preset(self, name: str, preset: CSliserPreset, *, include_sources: bool) -> None:
        settings = self._settings_factory()
        presets = self.load_presets(include_sources=include_sources)
        presets[name] = preset
        settings.setValue(
            _preset_key(include_sources),
            json.dumps(
                {key: value.to_legacy_dict(include_sources=include_sources) for key, value in presets.items()},
                ensure_ascii=False,
            ),
        )
        settings.sync()

    def delete_preset(self, name: str, *, include_sources: bool) -> None:
        settings = self._settings_factory()
        presets = self.load_presets(include_sources=include_sources)
        presets.pop(name, None)
        settings.setValue(
            _preset_key(include_sources),
            json.dumps(
                {key: value.to_legacy_dict(include_sources=include_sources) for key, value in presets.items()},
                ensure_ascii=False,
            ),
        )
        settings.sync()


def _preset_key(include_sources: bool) -> str:
    return "presets/all" if include_sources else "presets/frames"
