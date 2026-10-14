"""processing_mixin: flush pending editor frame."""
from __future__ import annotations
from time import perf_counter
from typing import Any
from contour.widget_parts._imports import (
    Path,
    PolygonData,
)


class FlushPendingEditorFrameMixin:
    def _apply_conductor_display_visibility(self: Any, state=None) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        filters_on = self._conductor_display_filters_enabled()
        if hasattr(self, "metal_show_rejected_checkbox"):
            layers: dict[str, list[PolygonData]] = {}
            if state is not None:
                layers = getattr(state, "metal_overlay_polygons", None) or {}
            elif self._workspace.current_state is not None:
                layers = getattr(self._workspace.current_state, "metal_overlay_polygons", None) or {}
            if filters_on:
                visibility = {
                    "rejected": self.metal_show_rejected_checkbox.isChecked(),
                    "suspicious": self.metal_show_suspicious_checkbox.isChecked(),
                    "border": self.metal_show_border_checkbox.isChecked(),
                    "wide_pairs_suspicious": self.metal_show_suspicious_checkbox.isChecked(),
                    "wide_pairs_rejected": self.metal_show_rejected_checkbox.isChecked(),
                }
            else:
                visibility = {
                    "rejected": False,
                    "suspicious": False,
                    "border": False,
                    "wide_pairs_suspicious": False,
                    "wide_pairs_rejected": False,
                }
            self.polygon_editor.set_metal_overlays(layers, visibility)
        show_conductors = True
        if filters_on and hasattr(self, "metal_show_conductors_checkbox"):
            show_conductors = self.metal_show_conductors_checkbox.isChecked()
        self.polygon_editor.set_polygon_category_visible("conductor", show_conductors)
        self.polygon_editor.set_polygon_category_visible("metal_border", show_conductors)
        self.polygon_editor.set_polygon_category_visible("metal_wide_gradient", show_conductors)

    def _flush_pending_editor_frame_apply(self: Any, image_path: str) -> None:
        pending = getattr(self, "_pending_editor_frame_apply", None)
        if pending is None:
            session = self._frame_switch_profile_for_path(image_path)
            if session is not None:
                session.complete_pending("editor_vectors", suffix="_none_pending")
            self._schedule_neighbor_frames_after_main_image_ready()
            return
        pending_path, polygons, defer_heavy_overlays = pending
        self._pending_editor_frame_apply = None
        if str(Path(pending_path)) != str(Path(image_path)):
            return
        if str(Path(pending_path)) != str(Path(self._workspace.current_image_path or "")):
            return
        current_state = self._workspace.current_state
        if current_state is None:
            return
        step_start = perf_counter()
        self._apply_editor_vectors_for_frame(
            pending_path,
            current_state,
            polygons,
            defer_heavy_overlays=defer_heavy_overlays,
        )
        if not defer_heavy_overlays:
            self._sync_extra_layers()
            self._refresh_gradient_overlay()
        else:
            self._schedule_deferred_frame_overlays(pending_path)
        self._update_vector_edit_status_label()
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.note_timing("editor_vectors_apply", (perf_counter() - step_start) * 1000.0)
            session.complete_pending("editor_vectors", suffix="_flushed")
        # Vectors just became visible (pixmap was async); validate off the UI thread.
        self._schedule_deferred_geometry_validation(pending_path)
        self._schedule_neighbor_frames_after_main_image_ready()
