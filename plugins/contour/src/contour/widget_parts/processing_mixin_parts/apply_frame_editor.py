"""processing_mixin: apply frame editor."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QPixmap,
)


class ApplyFrameEditorMixin:
    def _apply_frame_to_editor(
        self: Any,
        *,
        defer_neighbors: bool = True,
        defer_heavy_overlays: bool = False,
        preserve_view: bool = False,
        clear_neighbors: bool | None = None,
        sync_neighbors: bool = True,
    ) -> None:
        current_state = self._workspace.current_state
        image_path = self._workspace.current_image_path
        if clear_neighbors is None:
            clear_neighbors = not preserve_view
        if clear_neighbors:
            self._clear_neighbor_frame_display_for_frame_change()
        if current_state is None or not image_path:
            self._pending_editor_frame_apply = None
            self.polygon_editor.set_image_pixmap(QPixmap(), preserve_view=preserve_view)
            self.polygon_editor.set_polygons([], emit_signal=False)
            self._editor_polygons_signature = None
            self._last_editor_display_cache_key = None
            self._last_editor_display_path = None
            return
        display_image = self._display_image_for_current_state()
        polygons = list(current_state.polygons)
        display_cache_key = self._editor_display_cache_key(image_path, current_state)
        skip_display_refresh = bool(
            preserve_view
            and getattr(self, "_last_editor_display_cache_key", None) == display_cache_key
            and str(Path(getattr(self, "_last_editor_display_path", "") or "")) == str(Path(image_path))
        )
        self._editor_frame_pending_pixmap = None
        if skip_display_refresh:
            image_ready = True
        else:
            image_ready = self._apply_display_image_to_editor(
                image_path,
                display_image,
                state=current_state,
                preserve_view=preserve_view,
                apply_now=False,
            )
        pending_pixmap = getattr(self, "_editor_frame_pending_pixmap", None)
        self._editor_frame_pending_pixmap = None

        def _commit_layers() -> None:
            if pending_pixmap is not None:
                pixmap, pixmap_preserve_view, cache_key, path = pending_pixmap
                self.polygon_editor.set_image_pixmap(pixmap, preserve_view=pixmap_preserve_view)
                if cache_key is not None:
                    self._last_editor_display_cache_key = cache_key
                    self._last_editor_display_path = path
            self._apply_editor_vectors_for_frame(
                image_path,
                current_state,
                polygons,
                defer_heavy_overlays=defer_heavy_overlays,
            )

        if image_ready:
            self._pending_editor_frame_apply = None
            self._suspend_editor_viewport_updates(_commit_layers)
            if sync_neighbors:
                neighbor_delay_ms = 200 if self._uses_large_frame_list() else 0
                if defer_neighbors:
                    self._request_neighbor_frame_sync(delay_ms=neighbor_delay_ms)
                else:
                    self._request_neighbor_frame_sync(delay_ms=0)
            if not defer_heavy_overlays:
                self._sync_extra_layers()
                self._refresh_gradient_overlay()
            else:
                self._schedule_deferred_frame_overlays(image_path)
            self._update_vector_edit_status_label()
            return
        # Keep the previous complete frame on screen until the new pixmap is ready,
        # then swap image + vectors together in _on_display_ready.
        self._pending_editor_frame_apply = (
            str(Path(image_path)),
            polygons,
            defer_heavy_overlays,
        )
        session = self._frame_switch_profile_for_path(image_path)
        if session is not None:
            session.mark_pending("editor_vectors")
