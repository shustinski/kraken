"""processing_mixin: format invalid polygon description."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    summarize_invalid_polygon_description_reasons,
)


class FormatInvalidPolygonDescriptionMixin:
    def _format_invalid_polygon_description_reasons(
        self: Any,
        polygons: list,
        *,
        reasons_by_id: dict[int, list[str]] | None = None,
    ) -> str:
        reason_defaults = {
            "repeated_vertex": (
                "повтор вершины (связка keyhole)"
                if self._ui_language == "ru"
                else "repeated vertex (keyhole link)"
            ),
            "self_intersecting": (
                "самопересечение или самокасание контура"
                if self._ui_language == "ru"
                else "self-intersecting or self-touching outline"
            ),
            "overlapping": (
                "пересечение / наложение полигонов"
                if self._ui_language == "ru"
                else "intersecting / overlapping polygons"
            ),
            "small_object": (
                "площадь объекта ниже минимума"
                if self._ui_language == "ru"
                else "object area below minimum"
            ),
            "small_hole": (
                "площадь выреза ниже минимума"
                if self._ui_language == "ru"
                else "cutout area below minimum"
            ),
        }
        counts: dict[str, int] = {}
        order: list[str] = []
        if reasons_by_id is not None:
            for polygon in polygons:
                for reason in reasons_by_id.get(int(getattr(polygon, "id", -1)), ()):
                    if reason not in counts:
                        order.append(reason)
                        counts[reason] = 0
                    counts[reason] += 1
        else:
            settings = None
            if hasattr(self, "_vector_geometry_settings_from_widgets"):
                settings = self._vector_geometry_settings_from_widgets()
            for reason, count in summarize_invalid_polygon_description_reasons(polygons, settings):
                order.append(reason)
                counts[reason] = count
        parts: list[str] = []
        for reason in order:
            count = counts[reason]
            label = self._tr(
                f"invalid_polygon_reason_{reason}",
                reason_defaults.get(reason, reason),
            )
            if count > 1:
                parts.append(f"{label} ({count})")
            else:
                parts.append(label)
        if parts:
            return "; ".join(parts)
        return "—"
