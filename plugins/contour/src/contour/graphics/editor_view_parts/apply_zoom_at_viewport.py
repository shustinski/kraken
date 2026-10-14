"""editor_view: apply zoom at viewport."""
from __future__ import annotations
from math import hypot
from PyQt6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRectF,
)
from PyQt6.QtGui import (
    QResizeEvent,
)
from PyQt6.QtWidgets import QGraphicsView
from contour.graphics.geometry import (
    _snap_to_45,
)
from contour.graphics.minimap_geometry import MINIMAP_VIEWPORT_MARGIN_PX
from contour.graphics.tools import BrushMode
from contour.graphics.viewport_navigation import (
    clamp_zoom_factor,
    viewport_scroll_correction_after_scale_reanchor,
)

from contour.graphics.editor_view_parts.module_state import (
    _MINIMAP_DRAWING_TOOLS,
    _ZOOM_VECTOR_BATCH_THRESHOLD,
)


class ApplyZoomAtViewportMixin:
    def _enter_zoom_render_mode(self) -> None:
        self._editor_scene.begin_zoom_vector_render_mode(
            minimum_contacts=_ZOOM_VECTOR_BATCH_THRESHOLD,
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.NoViewportUpdate)
        self.setRenderHints(self._zooming_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)

    def _leave_zoom_render_mode(self) -> None:
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setRenderHints(self._steady_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)
        self._editor_scene.end_zoom_vector_render_mode()

    def _clamp_current_zoom_at_viewport_pixel(self, viewport_pixel: QPoint) -> None:
        current_zoom = self.zoom_factor()
        target_zoom = clamp_zoom_factor(current_zoom)
        if abs(target_zoom - current_zoom) <= 1e-9:
            return
        self._apply_zoom_at_viewport_pixel(viewport_pixel, target_zoom / current_zoom)

    def _apply_zoom_at_viewport_pixel(
        self,
        viewport_pixel: QPoint,
        factor: float,
        *,
        update_navigation: bool = True,
    ) -> None:
        if factor == 1.0 or factor <= 0:
            return
        view_point = self._viewport_to_view_point(viewport_pixel)
        scene_anchor = self.mapToScene(view_point)
        old_anchor = self.transformationAnchor()
        try:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            self.scale(factor, factor)
        finally:
            self.setTransformationAnchor(old_anchor)
        view_mapped = self.mapFromScene(scene_anchor)
        vp_mapped = self._view_to_viewport_point(view_mapped)
        dh, dv = viewport_scroll_correction_after_scale_reanchor(
            (viewport_pixel.x(), viewport_pixel.y()),
            (vp_mapped.x(), vp_mapped.y()),
        )
        h_scroll = self.horizontalScrollBar()
        v_scroll = self.verticalScrollBar()
        if h_scroll is not None:
            h_scroll.setValue(h_scroll.value() + dh)
        if v_scroll is not None:
            v_scroll.setValue(v_scroll.value() + dv)
        if update_navigation:
            self._update_navigation_scene_rect()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        if event is None:
            return
        super().resizeEvent(event)
        self._update_navigation_scene_rect()
        self._schedule_pyramid_visible_update()
        self._schedule_gradient_field_arrows_update()

    def leaveEvent(self, event: QEvent | None) -> None:
        if event is None:
            return
        self._editor_scene.clear_conductor_hover_highlight()
        self._editor_scene.clear_vertex_preview()
        self._editor_scene.clear_move_target_preview()
        self._editor_scene.hide_tool_cursors()
        super().leaveEvent(event)

    def _scene_tolerance(self, pixels: float | int) -> float:
        px = max(1, round(pixels))
        start = self.mapToScene(QPoint(0, 0))
        end = self.mapToScene(QPoint(px, 0))
        return max(1.0, abs(end.x() - start.x()))

    def _update_navigation_scene_rect(self, zoom: float | None = None) -> None:
        if self.pyramid_mode_enabled() and self._pyramid_layout is not None:
            base_rect = QRectF(self._pyramid_layout.scene_rect(self._pyramid_current_lod))
            base_rect = base_rect.united(QRectF(self._editor_scene.navigation_base_rect()))
        else:
            base_rect = QRectF(self._editor_scene.navigation_base_rect())
        if base_rect.width() <= 0.0 or base_rect.height() <= 0.0:
            self.setSceneRect(base_rect)
            self._update_minimap_overlay()
            return
        zoom = self.zoom_factor() if zoom is None else max(1e-6, float(zoom))
        viewport_rect = self._require_viewport().rect()
        margin_x = float(viewport_rect.width()) / max(zoom, 1e-6) + 2.0
        margin_y = float(viewport_rect.height()) / max(zoom, 1e-6) + 2.0
        self.setSceneRect(base_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y))
        self._update_minimap_overlay()

    def _refresh_minimap_thumbnail(self) -> None:
        pixmap = self._editor_scene.main_image_pixmap()
        image_rect = self._editor_scene.main_image_rect()
        self._minimap.set_image(pixmap, image_rect)
        self._update_minimap_overlay()

    def _position_minimap(self) -> None:
        viewport = self.viewport()
        if viewport is None:
            return
        viewport_geom = viewport.geometry()
        margin = int(MINIMAP_VIEWPORT_MARGIN_PX)
        x = viewport_geom.right() - self._minimap.width() - margin
        y = viewport_geom.bottom() - self._minimap.height() - margin
        self._minimap.move(max(viewport_geom.left() + margin, x), max(viewport_geom.top() + margin, y))

    def _update_minimap_overlay(self) -> None:
        if not self._minimap.has_image():
            self._minimap.hide()
            return
        self._minimap.show()
        self._minimap.raise_()
        self._position_minimap()
        self._update_minimap_mouse_interaction()
        viewport = self.viewport()
        if viewport is None:
            return
        viewport_scene = self.mapToScene(viewport.rect()).boundingRect()
        self._minimap.set_viewport_scene_rect(viewport_scene)

    def _update_minimap_mouse_interaction(self) -> None:
        self._minimap.set_mouse_interactive(self._tool not in _MINIMAP_DRAWING_TOOLS)

    def _on_minimap_scene_point(self, scene_pos: QPointF) -> None:
        self.centerOn(scene_pos)
        self._update_navigation_scene_rect()

    def _on_minimap_scene_delta(self, delta: QPointF) -> None:
        viewport = self.viewport()
        if viewport is None:
            return
        center = self.mapToScene(viewport.rect().center())
        self.centerOn(QPointF(center.x() + delta.x(), center.y() + delta.y()))
        self._update_navigation_scene_rect()

    def _append_brush_point(self, scene_pos: QPointF) -> None:
        target = scene_pos
        if self._brush_mode == BrushMode.ANGLED:
            last_point = self._editor_scene.pending_last_point()
            if last_point is not None:
                target = _snap_to_45(last_point, scene_pos)
        last_point = self._editor_scene.pending_last_point()
        if last_point is not None and hypot(target.x() - last_point.x(), target.y() - last_point.y()) < 1.0:
            # Brush vertex spacing rule: at least 1 image pixel in scene/image coordinates.
            return
        self._editor_scene.append_brush_vertex(target, self._brush_thickness)

    def _start_rectangle_polygon(self, scene_pos: QPointF, *, erase: bool) -> None:
        self._drag_kind = "rect_polygon"
        self._drag_start_scene_pos = QPointF(scene_pos)
        self._drag_erases = bool(erase)
        self._editor_scene.set_preview_rect(scene_pos, scene_pos)

    def _commit_rectangle_polygon(self, scene_pos: QPointF) -> None:
        start = self._drag_start_scene_pos
        erase = self._drag_erases
        self._drag_kind = None
        self._drag_start_scene_pos = None
        self._drag_erases = False
        if start is None:
            self._editor_scene.clear_preview_rect()
            return
        self._editor_scene.add_rectangle_polygon(start, scene_pos, erase=erase)

    def _cancel_rectangle_polygon(self) -> None:
        if self._drag_kind != "rect_polygon":
            return
        self._drag_kind = None
        self._drag_start_scene_pos = None
        self._drag_erases = False
        self._editor_scene.clear_preview_rect()
