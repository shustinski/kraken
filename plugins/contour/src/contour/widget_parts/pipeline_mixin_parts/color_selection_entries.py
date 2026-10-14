"""pipeline_mixin: color selection entries."""
from __future__ import annotations


class ColorSelectionEntriesMixin:
    def _update_step_parameter(self, row: int, parameter_name: str, value) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters[parameter_name] = value
        self._auto_apply_pipeline()

    def _color_selection_entries(self, row: int) -> list[ColorSelectionEntry]:
        if row < 0 or row >= len(self._pipeline.steps):
            return []
        entries = self._pipeline.steps[row].parameters.get("selected_colors", [])
        if not isinstance(entries, list):
            entries = []
        normalized: list[ColorSelectionEntry] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rgb = entry.get("rgb")
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                continue
            try:
                parsed_rgb = [max(0, min(255, int(channel))) for channel in rgb]
            except (TypeError, ValueError):
                continue
            normalized.append({"rgb": parsed_rgb, "enabled": bool(entry.get("enabled", True))})
        self._pipeline.steps[row].parameters["selected_colors"] = normalized
        return normalized
