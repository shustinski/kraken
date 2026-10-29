"""processing_mixin: apply display image editor."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QPixmap,
    QTimer,
)


class ApplyDisplayImageEditorMixin:
    def _ensure_source_preview_pixmap_cached(self: Any, image_path: str, state) -> None:
        if state is None or state.source_image is None:
            return
        source_key = (str(Path(image_path)), "source", "")
        if source_key in getattr(self, "_editor_pixmap_cache", {}):
            return
        pending_keys = getattr(self, "_editor_source_cache_pending_keys", None)
        if pending_keys is None:
            pending_keys = set()
            self._editor_source_cache_pending_keys = pending_keys
        if source_key in pending_keys:
            return
        pending_keys.add(source_key)
        self._queue_editor_display_pixmap(
            image_path,
            state.source_image,
            cache_key=source_key,
            preserve_view=True,
            cache_only=True,
        )

    def _apply_display_image_to_editor(
        self: Any,
        image_path: str,
        display_image: object,
        *,
        state,
        preserve_view: bool = False,
        apply_now: bool = True,
    ) -> bool:
        if display_image is None:
            if apply_now:
                self.polygon_editor.set_image_pixmap(QPixmap(), preserve_view=preserve_view)
            else:
                self._editor_frame_pending_pixmap = (QPixmap(), preserve_view, None, str(Path(image_path)))
            return True
        cache_key = self._editor_display_cache_key(image_path, state)
        cached = getattr(self, "_editor_pixmap_cache", {}).get(cache_key)
        if cached is not None and not cached.isNull():
            if apply_now:
                self.polygon_editor.set_image_pixmap(cached, preserve_view=preserve_view)
                self._last_editor_display_cache_key = cache_key
                self._last_editor_display_path = str(Path(image_path))
            else:
                self._editor_frame_pending_pixmap = (
                    cached,
                    preserve_view,
                    cache_key,
                    str(Path(image_path)),
                )
            if cache_key[1] != "source":
                self._ensure_source_preview_pixmap_cached(image_path, state)
            return True
        self._editor_frame_pending_pixmap = None
        self._queue_editor_display_pixmap(
            image_path,
            display_image,
            cache_key=cache_key,
            preserve_view=preserve_view,
        )
        if cache_key[1] != "source":
            self._ensure_source_preview_pixmap_cached(image_path, state)
        return False

    def _refresh_current_display_image_only(self: Any, *, preserve_view: bool = True) -> None:
        current_state = self._workspace.current_state
        image_path = self._workspace.current_image_path
        if current_state is None or not image_path:
            self.polygon_editor.set_image_pixmap(QPixmap(), preserve_view=preserve_view)
            self._last_editor_display_cache_key = None
            self._last_editor_display_path = None
            return
        display_image = self._display_image_for_current_state()
        image_ready = self._apply_display_image_to_editor(
            image_path,
            display_image,
            state=current_state,
            preserve_view=preserve_view,
        )
        if image_ready:
            self._last_editor_display_cache_key = self._editor_display_cache_key(image_path, current_state)
            self._last_editor_display_path = str(Path(image_path))

    def _recognition_hides_loaded_vectors(self: Any) -> bool:
        if not hasattr(self, "recognition_mode_combo"):
            return False
        return str(self.recognition_mode_combo.currentData() or "") != "disabled"

    def _state_has_recognition_result(self: Any, state) -> bool:
        if state is None:
            return False
        if getattr(state, "recognition_produced", False):
            return True
        if getattr(state, "mask_image", None) is not None:
            return True
        if getattr(state, "metal_overlay_polygons", None):
            return True
        return bool(getattr(state, "recognition_base_polygons", None))

    def _polygons_visible_in_editor(self: Any, state, polygons: list) -> list:
        if not self._recognition_hides_loaded_vectors():
            return list(polygons)
        if self._state_has_recognition_result(state):
            return list(polygons)
        return []

    def _apply_editor_vectors_for_frame(
        self: Any,
        image_path: str,
        state,
        polygons: list,
        *,
        defer_heavy_overlays: bool,
    ) -> None:
        visible_polygons = self._polygons_visible_in_editor(state, polygons)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_polygon_overlays_visible(True)
        self._sync_polygons_to_editor(image_path, visible_polygons)
        if defer_heavy_overlays:
            return
        self._apply_deferred_editor_vector_overlays(state)

    def _apply_deferred_editor_vector_overlays(self: Any, state) -> None:
        debug_candidates = list(state.debug_candidates)
        if (
            not hasattr(self, "debug_candidates_checkbox")
            or not self.debug_candidates_checkbox.isChecked()
        ):
            debug_candidates = []
        elif (
            hasattr(self, "bright_via_show_rejected_checkbox")
            and not self.bright_via_show_rejected_checkbox.isChecked()
        ):
            debug_candidates = [
                candidate
                for candidate in debug_candidates
                if bool(getattr(candidate, "accepted", False))
            ]
        self.polygon_editor.set_debug_candidates(debug_candidates)
        self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
        if hasattr(self, "via_show_detected_checkbox"):
            self.polygon_editor.set_polygon_category_visible(
                "via", self.via_show_detected_checkbox.isChecked()
            )
        self._apply_conductor_display_visibility(state)

    def _schedule_deferred_frame_overlays(self: Any, image_path: str) -> None:
        def _apply() -> None:
            if bool(getattr(self, "_closing", False)):
                return
            if str(Path(self._workspace.current_image_path or "")) != str(Path(image_path)):
                return
            state = self._workspace.current_state
            if state is None:
                return
            self._apply_deferred_editor_vector_overlays(state)
            self._sync_extra_layers()
            self._refresh_gradient_overlay()

        QTimer.singleShot(0, _apply)

    def _conductor_display_filters_enabled(self: Any) -> bool:
        return (
            hasattr(self, "recognition_mode_combo")
            and str(self.recognition_mode_combo.currentData() or "") == "conductors"
        )
