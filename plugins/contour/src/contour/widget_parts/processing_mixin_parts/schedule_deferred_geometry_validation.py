"""processing_mixin: schedule deferred geometry validation."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
)


class ScheduleDeferredGeometryValidationMixin:
    def _offer_invalid_polygon_descriptions_if_vectors_visible(self: Any, image_path: str) -> None:
        """Start deferred geometry validation once current-frame vectors are on the editor."""

        if getattr(self, "_pending_editor_frame_apply", None) is not None:
            return
        if not self._editor_polygons_are_current_frame():
            return
        self._schedule_deferred_geometry_validation(image_path)

    def _schedule_deferred_geometry_validation(self: Any, image_path: str) -> None:
        """Validate repair reasons off the UI thread; abort if the frame switches."""

        if bool(getattr(self, "_closing", False)):
            return
        expected = str(Path(image_path))
        if getattr(self, "_frame_switch_profile", None) is not None:
            self._pending_geometry_validation_path = expected
            return
        self._start_deferred_geometry_validation(expected)
