"""processing_mixin: start deferred geometry validation."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    GeometryValidationRunnable,
    Path,
)


class StartDeferredGeometryValidationMixin:
    def _start_deferred_geometry_validation(self: Any, image_path: str) -> None:
        if bool(getattr(self, "_closing", False)):
            return
        if getattr(self, "_pending_editor_frame_apply", None) is not None:
            return
        if not self._editor_polygons_are_current_frame():
            return
        expected = str(Path(image_path))
        current = str(Path(self._workspace.current_image_path or ""))
        if expected != current:
            return
        cached = self._repair_reasons_for_image_path(expected)
        if cached is not None:
            if hasattr(self, "polygon_editor") and hasattr(self.polygon_editor, "apply_polygons_needing_repair"):
                self.polygon_editor.apply_polygons_needing_repair(cached)
            if cached:
                self._schedule_invalid_polygon_description_offer(expected)
            return
        self._geometry_validation_generation = int(getattr(self, "_geometry_validation_generation", 0)) + 1
        generation = self._geometry_validation_generation
        state = self._workspace.cached_state(expected)
        polygons = list(state.polygons) if state is not None else self.polygon_editor.get_polygons()
        if not polygons:
            if hasattr(self, "polygon_editor") and hasattr(self.polygon_editor, "apply_polygons_needing_repair"):
                self.polygon_editor.apply_polygons_needing_repair({})
            return
        profile_session = self._frame_switch_profile_for_path(expected)
        runnable = GeometryValidationRunnable(
            generation,
            expected,
            polygons,
            self._make_polygon_repair_scanner(include_ring_reasons=False),
            profile_session=profile_session,
        )

        def _on_result(req_id: int, path: str, reasons_obj: dict[int, list[str]] | None) -> None:
            if bool(getattr(self, "_closing", False)):
                return
            if req_id != int(getattr(self, "_geometry_validation_generation", 0)):
                return
            if str(Path(self._workspace.current_image_path or "")) != str(Path(path)):
                return
            reasons = {
                int(polygon_id): list(codes)
                for polygon_id, codes in dict(reasons_obj or {}).items()
            }
            cached_state = self._workspace.cached_state(path)
            if cached_state is not None:
                cached_state.polygons_needing_repair = reasons
            if hasattr(self, "polygon_editor") and hasattr(self.polygon_editor, "apply_polygons_needing_repair"):
                self.polygon_editor.apply_polygons_needing_repair(reasons)
            if reasons:
                self._schedule_invalid_polygon_description_offer(path)

        def _on_error(req_id: int, path: str, message: str) -> None:
            if req_id != int(getattr(self, "_geometry_validation_generation", 0)):
                return
            if str(Path(self._workspace.current_image_path or "")) != str(Path(path)):
                return
            self._append_log(
                self._tr(
                    "geometry_validation_failed_log",
                    "Geometry validation failed: {error}" if self._ui_language != "ru" else "Проверка геометрии не удалась: {error}",
                    error=message,
                )
            )

        runnable.signals.result.connect(_on_result)
        runnable.signals.error.connect(_on_error)
        self._frame_load_thread_pool.start(runnable)
