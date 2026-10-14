"""editor_view: pyramid frame store."""
from __future__ import annotations
from math import log2
from PyQt6.QtCore import (
    QRectF,
)
from PyQt6.QtGui import (
    QPixmap,
)
from ...application.frame_lod import FixedGridFrameLayout, PyramidFrameStore
from contour.domain import PolygonData


class PyramidFrameStoreMixin:
    def apply_polygons_needing_repair(
        self,
        reasons: dict[int, list[str]] | None,
        *,
        refresh_items: bool = True,
    ) -> None:
        self._editor_scene.apply_polygons_needing_repair(reasons, refresh_items=refresh_items)

    def polygons_needing_repair_map(self) -> dict[int, list[str]]:
        return self._editor_scene.polygons_needing_repair_map()

    def refresh_polygon_overlays(self) -> None:
        self._editor_scene.refresh_polygon_items()

    def get_polygons(self) -> list[PolygonData]:
        return self._editor_scene.get_polygons()

    def antialias_selected_polygons(self, grade: int) -> bool:
        return self._editor_scene.antialias_selected_polygons(grade)

    def repair_invalid_polygon_descriptions(self) -> bool:
        return self._editor_scene.repair_invalid_polygon_descriptions()

    def set_antialias_grade(self, grade: int) -> None:
        self._antialias_grade = max(1, int(grade))

    def set_neighbor_frames(
        self,
        frames: list[tuple],
        opacity: float,
        overlap_pixels: int = 0,
        show_main_frame: bool = True,
    ) -> None:
        self._editor_scene.set_neighbor_frames(frames, opacity, overlap_pixels, show_main_frame)
        self._refresh_neighbor_viewport()

    def _refresh_neighbor_viewport(self) -> None:
        """Expand scroll range for the neighbor grid without changing pan/zoom."""
        self._update_navigation_scene_rect()
        self._editor_scene.update(self._editor_scene.sceneRect())
        viewport = self.viewport()
        if viewport is not None:
            viewport.update()

    def set_debug_candidates(self, candidates: list[object]) -> None:
        self._editor_scene.set_debug_candidates(candidates)

    def set_metal_overlays(self, layers: dict[str, list[PolygonData]], visibility: dict[str, bool]) -> None:
        self._editor_scene.set_metal_overlays(layers, visibility)

    def set_via_debug_inspection_enabled(self, enabled: bool) -> None:
        self._via_debug_inspection_enabled = bool(enabled)
        self._editor_scene.set_debug_candidates([])

    def zoom_factor(self) -> float:
        return max(1e-6, float(self.transform().m11()))

    def set_pyramid_frame_store(
        self,
        store: PyramidFrameStore | None,
        *,
        frame_count: int | None = None,
        columns: int | None = None,
        current_frame_id: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Enable a virtualized multi-frame pyramid display when a store is available."""

        self._clear_pyramid_items()
        self._pyramid_generation += 1
        self._pyramid_store = store
        self._pyramid_layout = None
        self._pyramid_pixmap_cache.clear()
        self._pyramid_pending_loads.clear()
        if store is None:
            self._pyramid_enabled = False
            self._pyramid_current_frame_id = None
            self._pyramid_selection_item.hide()
            self._update_navigation_scene_rect()
            return
        count = max(0, int(frame_count if frame_count is not None else store.frame_count()))
        should_enable = bool(store.has_lod()) if enabled is None else bool(enabled and store.has_lod())
        if count <= 0 or not should_enable:
            self._pyramid_enabled = False
            self._pyramid_current_frame_id = current_frame_id
            self._pyramid_selection_item.hide()
            self._update_navigation_scene_rect()
            return
        if columns is None:
            columns = max(1, round(count ** 0.5))
        self._pyramid_layout = FixedGridFrameLayout(
            frame_count=count,
            columns=max(1, int(columns)),
            frame_store=store,
            gap=16,
        )
        self._pyramid_enabled = True
        self._pyramid_current_lod = self.choose_lod(self.zoom_factor(), store.max_lod())
        self.set_current_frame_id(0 if current_frame_id is None else current_frame_id, center=False, emit_signal=False)
        self._update_navigation_scene_rect()
        self._schedule_pyramid_visible_update()

    def set_pyramid_frames(
        self,
        store: PyramidFrameStore | None,
        *,
        frame_count: int | None = None,
        columns: int | None = None,
        current_frame_id: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.set_pyramid_frame_store(
            store,
            frame_count=frame_count,
            columns=columns,
            current_frame_id=current_frame_id,
            enabled=enabled,
        )

    def pyramid_mode_enabled(self) -> bool:
        return bool(self._pyramid_enabled and self._pyramid_store is not None and self._pyramid_layout is not None)

    def choose_lod(self, zoom: float, max_lod: int) -> int:
        max_lod = max(0, int(max_lod))
        zoom = max(1e-6, float(zoom))
        target = max(0, min(max_lod, round(log2(1.0 / zoom))))
        current = max(0, min(max_lod, int(getattr(self, "_pyramid_current_lod", 0))))
        if target == current:
            return current
        # Hysteresis keeps the pyramid from swapping LODs repeatedly near 2x boundaries.
        if target > current:
            switch_zoom = (2.0 ** (-(current + 0.65)))
            return target if zoom < switch_zoom else current
        switch_zoom = (2.0 ** (-(current - 0.35)))
        return target if zoom > switch_zoom else current

    def current_frame_id(self) -> int | None:
        return self._pyramid_current_frame_id

    def set_current_frame_id(self, frame_id: int | None, *, center: bool = True, emit_signal: bool = True) -> None:
        layout = self._pyramid_layout
        if frame_id is None:
            self._pyramid_current_frame_id = None
            self._pyramid_selection_item.hide()
            return
        frame_id = int(frame_id)
        if layout is not None:
            frame_id = max(0, min(layout.frame_count - 1, frame_id))
        changed = frame_id != self._pyramid_current_frame_id
        self._pyramid_current_frame_id = frame_id
        self._update_pyramid_selection_rect()
        if center:
            self.center_on_frame(frame_id)
        if changed and emit_signal:
            self.currentFrameChanged.emit(frame_id)

    def center_on_frame(self, frame_id: int | None) -> None:
        if frame_id is None:
            return
        layout = self._pyramid_layout
        if layout is None:
            self.center_main_image()
            return
        rect = layout.frame_id_to_scene_rect(int(frame_id), self._pyramid_current_lod)
        if rect.width() > 0 and rect.height() > 0:
            self.centerOn(rect.center())
            self._update_navigation_scene_rect()
            self._schedule_pyramid_visible_update()

    def _clear_pyramid_items(self) -> None:
        self._pyramid_visible_timer.stop()
        for item in self._pyramid_visible_items.values():
            item.setPixmap(QPixmap())
            if item.scene() is not None:
                self._editor_scene.removeItem(item)
        self._pyramid_visible_items.clear()
        self._pyramid_pixmap_cache.clear()

    def _pyramid_viewport_scene_rect(self) -> QRectF:
        viewport = self._require_viewport().rect()
        polygon = self.mapToScene(viewport)
        return polygon.boundingRect()

    def _clear_gradient_field_maps(self) -> None:
        self._gradient_arrows_timer.stop()
        self._gradient_field_x = None
        self._gradient_field_y = None
        self._gradient_field_peak = 0.0
        self._editor_scene.clear_gradient_field_arrows()

    def _schedule_gradient_field_arrows_update(self, *_args) -> None:
        if self._gradient_field_x is None or self._gradient_field_y is None:
            return
        self._gradient_arrows_timer.stop()
        self._gradient_arrows_timer.start()
