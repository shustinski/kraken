"""processing_mixin: schedule invalid polygon description."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QTimer,
)


class ScheduleInvalidPolygonDescriptionMixin:
    def _schedule_invalid_polygon_description_offer(self: Any, image_path: str) -> None:
        """Prompt for topology repair after deferred validation marks invalid polygons."""

        if bool(getattr(self, "_invalid_polygon_offer_active", False)):
            return
        expected = str(Path(image_path))
        editor = getattr(self, "polygon_editor", None)
        if editor is not None:
            viewport = getattr(editor, "viewport", None)
            view = viewport() if callable(viewport) else None
            if view is not None:
                view.repaint()
        QTimer.singleShot(
            0,
            lambda path=expected: self._offer_repair_invalid_polygon_descriptions(path),
        )
