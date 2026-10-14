"""processing_mixin: contact center."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    PolygonData,
)


class ContactCenterMixin:
    def _contact_feedback_text(self: Any, ru: str, en: str) -> str:
        return ru if getattr(self, "_ui_language", "en") == "ru" else en

    @staticmethod
    def _contact_center(polygon: PolygonData) -> tuple[float, float]:
        x_coord, y_coord, width, height = polygon.bbox
        if width > 0 and height > 0:
            return x_coord + 0.5 * width, y_coord + 0.5 * height
        if not polygon.points:
            return 0.0, 0.0
        return (
            sum(float(point[0]) for point in polygon.points) / len(polygon.points),
            sum(float(point[1]) for point in polygon.points) / len(polygon.points),
        )
