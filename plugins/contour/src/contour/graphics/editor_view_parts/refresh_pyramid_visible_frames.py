"""editor_view: refresh pyramid visible frames."""
from __future__ import annotations
from PyQt6.QtCore import (
    QPoint,
    Qt,
)
from PyQt6.QtGui import (
    QPixmap,
)
from PyQt6.QtWidgets import QGraphicsPixmapItem
from ...adapters.qt.pyramid import PyramidFrameLoadRunnable
from ...application.processing import DisplaySettings


class RefreshPyramidVisibleFramesMixin:
    def _refresh_gradient_field_arrows(self) -> None:
        gradient_x = self._gradient_field_x
        gradient_y = self._gradient_field_y
        if gradient_x is None or gradient_y is None:
            self._editor_scene.clear_gradient_field_arrows()
            return
        from contour.graphics.gradient_field_arrows import sample_gradient_field_arrows

        visible = self._pyramid_viewport_scene_rect()
        zoom = self.zoom_factor()
        scene_units_per_view_px = 1.0 / max(zoom, 1e-12)
        arrows = sample_gradient_field_arrows(
            gradient_x,
            gradient_y,
            (visible.left(), visible.top(), visible.right(), visible.bottom()),
            scene_units_per_view_px,
            peak_magnitude=self._gradient_field_peak,
        )
        self._editor_scene.set_gradient_field_arrows(arrows)

    def _schedule_pyramid_visible_update(self) -> None:
        if not self.pyramid_mode_enabled():
            return
        self._pyramid_visible_timer.stop()
        self._pyramid_visible_timer.start()

    def _refresh_pyramid_visible_frames(self) -> None:
        store = self._pyramid_store
        layout = self._pyramid_layout
        if store is None or layout is None or not self._pyramid_enabled:
            return
        new_lod = self.choose_lod(self.zoom_factor(), store.max_lod())
        if new_lod != self._pyramid_current_lod:
            self._pyramid_current_lod = new_lod
            self._clear_pyramid_items()
            self._update_navigation_scene_rect()
            self._update_pyramid_selection_rect()
        viewport_rect = self._pyramid_viewport_scene_rect()
        if store.max_lod() <= 0 and self._pyramid_current_lod == 0:
            visible = set()
        else:
            visible = set(layout.frame_ids_intersecting(viewport_rect, self._pyramid_current_lod, buffer_cells=0))
        if self._pyramid_current_frame_id is not None:
            visible.add(int(self._pyramid_current_frame_id))
        for frame_id in list(self._pyramid_visible_items):
            if frame_id in visible:
                continue
            item = self._pyramid_visible_items.pop(frame_id)
            item.setPixmap(QPixmap())
            if item.scene() is not None:
                self._editor_scene.removeItem(item)
            for cache_key in list(self._pyramid_pixmap_cache):
                if cache_key[0] == frame_id:
                    self._pyramid_pixmap_cache.pop(cache_key, None)
        self._prune_pyramid_pixmap_cache(visible)
        for frame_id in sorted(visible):
            self._ensure_pyramid_frame_item(frame_id)
        self._update_pyramid_selection_rect()
        self.editorViewportChanged.emit(viewport_rect)

    def _ensure_pyramid_frame_item(self, frame_id: int) -> None:
        layout = self._pyramid_layout
        if layout is None:
            return
        key = (int(frame_id), int(self._pyramid_current_lod))
        item = self._pyramid_visible_items.get(frame_id)
        if item is None:
            item = QGraphicsPixmapItem()
            item.setZValue(-30)
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self._editor_scene.addItem(item)
            self._pyramid_visible_items[frame_id] = item
        rect = layout.frame_id_to_scene_rect(frame_id, self._pyramid_current_lod)
        item.setPos(rect.topLeft())
        pixmap = self._pyramid_cached_pixmap(key)
        if pixmap is not None and not pixmap.isNull():
            item.setPixmap(pixmap)
            item.setScale(rect.width() / max(1, pixmap.width()))
            item.show()
            return
        item.hide()
        self._queue_pyramid_frame_load(frame_id, self._pyramid_current_lod)

    def _pyramid_cached_pixmap(self, key: tuple[int, int]) -> QPixmap | None:
        pixmap = self._pyramid_pixmap_cache.get(key)
        if pixmap is not None:
            self._pyramid_pixmap_cache.move_to_end(key)
        return pixmap

    def _cache_pyramid_pixmap(self, key: tuple[int, int], pixmap: QPixmap) -> None:
        if key[0] not in self._pyramid_visible_items:
            return
        self._pyramid_pixmap_cache[key] = pixmap
        self._pyramid_pixmap_cache.move_to_end(key)
        self._prune_pyramid_pixmap_cache(set(self._pyramid_visible_items))

    def _prune_pyramid_pixmap_cache(self, visible_frame_ids: set[int]) -> None:
        visible_keys = {(int(frame_id), int(self._pyramid_current_lod)) for frame_id in visible_frame_ids}
        for key in list(self._pyramid_pixmap_cache):
            if key not in visible_keys:
                self._pyramid_pixmap_cache.pop(key, None)

    def _queue_pyramid_frame_load(self, frame_id: int, lod: int) -> None:
        store = self._pyramid_store
        if store is None:
            return
        key = (int(frame_id), int(lod))
        if key in self._pyramid_pending_loads:
            return
        self._pyramid_pending_loads.add(key)
        generation = self._pyramid_generation
        runnable = PyramidFrameLoadRunnable(generation, frame_id, lod, store)
        runnable.signals.result.connect(self._on_pyramid_frame_loaded)
        runnable.signals.error.connect(self._on_pyramid_frame_error)
        self._pyramid_thread_pool.start(runnable)

    def _on_pyramid_frame_loaded(self, generation: int, frame_id: int, lod: int, qimage: object) -> None:
        key = (int(frame_id), int(lod))
        self._pyramid_pending_loads.discard(key)
        if int(generation) != int(self._pyramid_generation):
            return
        pixmap = QPixmap.fromImage(qimage) if hasattr(qimage, "isNull") and not qimage.isNull() else QPixmap()
        if pixmap.isNull():
            return
        self._cache_pyramid_pixmap(key, pixmap)
        if lod == self._pyramid_current_lod and frame_id in self._pyramid_visible_items:
            self._ensure_pyramid_frame_item(int(frame_id))

    def _on_pyramid_frame_error(self, generation: int, frame_id: int, lod: int, message: str) -> None:
        self._pyramid_pending_loads.discard((int(frame_id), int(lod)))
        if int(generation) == int(self._pyramid_generation):
            self.logRequested.emit(f"[contour pyramid] frame={frame_id} lod={lod} load failed: {message}")

    def _update_pyramid_selection_rect(self) -> None:
        layout = self._pyramid_layout
        frame_id = self._pyramid_current_frame_id
        if not self._pyramid_enabled or layout is None or frame_id is None:
            self._pyramid_selection_item.hide()
            return
        rect = layout.frame_id_to_scene_rect(int(frame_id), self._pyramid_current_lod)
        self._pyramid_selection_item.setRect(rect.adjusted(-2.0, -2.0, 2.0, 2.0))
        self._pyramid_selection_item.show()

    def _pyramid_frame_at_viewport_pos(self, viewport_pos: QPoint) -> int | None:
        layout = self._pyramid_layout
        if not self.pyramid_mode_enabled() or layout is None:
            return None
        scene_pos = self.mapToScene(self._viewport_to_view_point(viewport_pos))
        return layout.scene_pos_to_frame_id(scene_pos.x(), scene_pos.y(), self._pyramid_current_lod)

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._editor_scene.set_display_settings(settings)
        self._update_tool_cursors()

    def set_random_object_colors_enabled(self, enabled: bool) -> None:
        self._editor_scene.set_random_object_colors_enabled(enabled)

    def flush_deferred_object_colors(self) -> None:
        self._editor_scene.flush_deferred_object_colors()

    def set_extra_layers(self, layers: list[dict[str, object]]) -> None:
        self._editor_scene.set_extra_layers(layers)

    def set_gradient_overlay(self, image, opacity: float = 0.45) -> None:
        self._clear_gradient_field_maps()
        self._editor_scene.set_gradient_overlay(image, opacity)

    def clear_gradient_overlay(self) -> None:
        self._clear_gradient_field_maps()
        self._editor_scene.clear_gradient_overlay()

    def set_gradient_field_maps(self, gradient_x, gradient_y) -> None:
        self._gradient_field_x = gradient_x
        self._gradient_field_y = gradient_y
        self._gradient_field_peak = 0.0
        if gradient_x is not None and gradient_y is not None:
            from contour.graphics.gradient_field_arrows import peak_gradient_magnitude

            self._gradient_field_peak = peak_gradient_magnitude(gradient_x, gradient_y)
        self._refresh_gradient_field_arrows()

    def set_gradient_overlay_opacity(self, opacity: float) -> None:
        self._editor_scene.set_gradient_overlay_opacity(opacity)

    def set_polygon_category_visible(self, category: str, visible: bool) -> None:
        self._editor_scene.set_polygon_category_visible(category, visible)

    def set_polygon_overlays_visible(self, visible: bool) -> None:
        if self._polygon_overlay_hide_holds:
            self._polygon_overlays_visible_before_holds = bool(visible)
            self._editor_scene.set_polygon_overlays_visible(False)
            return
        self._editor_scene.set_polygon_overlays_visible(visible)

    def polygon_overlays_visible(self) -> bool:
        return self._editor_scene.polygon_overlays_visible()

    def set_ui_language(self, language: str | None) -> None:
        self._editor_scene.set_ui_language(language)

    def set_image_click_mode(self, enabled: bool) -> None:
        self._image_click_mode = bool(enabled)
