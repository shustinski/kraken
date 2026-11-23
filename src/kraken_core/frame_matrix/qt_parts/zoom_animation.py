"""Smooth zoom under the cursor, using the same timing as Contour's matrix."""

from __future__ import annotations

import math

from time import perf_counter

from PyQt6.QtCore import QPoint, QRectF, Qt, QTimer
from PyQt6.QtGui import QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsView, QStyleOptionGraphicsItem

from kraken_core.frame_matrix import profiling

_ZOOM_FRAME_MS = 16
_ZOOM_EASING = 0.55
_ZOOM_SETTLE = 0.001
_WHEEL_ZOOM_STEP = 1.28


def wheel_zoom_factor(delta_y: float) -> float:
    """One wheel notch of 120 units multiplies the zoom by 1.28."""

    if delta_y == 0:
        return 1.0
    return _WHEEL_ZOOM_STEP ** (float(delta_y) / 120.0)


class ZoomAnimationMixin:
    def _configure_matrix_viewport(self) -> None:
        self._zoom_animating = False
        self._capturing_zoom_frame = False
        self._zoom_animation_target = 1.0
        self._zoom_animation_anchor: QPoint | None = None
        self._matrix_zoom_profile = None
        self._zoom_animation_timer = QTimer(self)
        self._zoom_animation_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._zoom_animation_timer.setInterval(_ZOOM_FRAME_MS)
        self._zoom_animation_timer.timeout.connect(self._advance_zoom_animation)
        self._steady_render_hints = QPainter.RenderHint.SmoothPixmapTransform
        self._zooming_render_hints = QPainter.RenderHint(0)
        self.setRenderHints(self._steady_render_hints)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)

    def _start_zoom_animation(self, viewport_pixel: QPoint, factor: float) -> None:
        base = self._zoom_animation_target if self._zoom_animation_timer.isActive() else self.zoom_factor()
        target = min(self.MAX_ZOOM, max(self._fitted_zoom_factor(), base * float(factor)))
        if math.isclose(target, self.zoom_factor(), rel_tol=0.0, abs_tol=1e-9):
            return
        self._track_zoom_profile(self.zoom_factor(), target)
        self._zoom_animation_target = target
        self._zoom_animation_anchor = QPoint(viewport_pixel)
        if not self._zoom_animating:
            self._enter_zoom_render_mode()
        self._advance_zoom_animation()
        if self._zoom_animation_anchor is not None and not self._zoom_animation_timer.isActive():
            self._zoom_animation_timer.start()

    def _track_zoom_profile(self, initial_zoom: float, target_zoom: float) -> None:
        profile = self._matrix_zoom_profile
        if profile is not None:
            profile.update_target(target_zoom)
            return
        self._matrix_zoom_profile = profiling.begin_matrix_zoom(initial_zoom, target_zoom)

    def _finish_zoom_profile(self, status: str) -> None:
        profile = self._matrix_zoom_profile
        if profile is None:
            return
        self._matrix_zoom_profile = None
        profile.finish(status=status, final_zoom=self.zoom_factor())

    def _advance_zoom_animation(self) -> None:
        frame_started_at = perf_counter()
        anchor = self._zoom_animation_anchor
        if anchor is None:
            if self._matrix_zoom_profile is not None:
                self._matrix_zoom_profile.record_frame(frame_started_at)
            self._stop_zoom_animation(rebuild=True)
            return
        current = self.zoom_factor()
        remaining = self._zoom_animation_target - current
        settle = max(_ZOOM_SETTLE, abs(self._zoom_animation_target) * _ZOOM_SETTLE)
        if abs(remaining) <= settle:
            next_zoom = self._zoom_animation_target
            finish = True
        else:
            next_zoom = current + remaining * _ZOOM_EASING
            finish = False
        if current > 0:
            self._apply_zoom_at_viewport_pixel(anchor, next_zoom / current)
        if self._matrix_zoom_profile is not None:
            self._matrix_zoom_profile.record_frame(frame_started_at)
        if finish:
            self._stop_zoom_animation(rebuild=True)

    def _apply_zoom_at_viewport_pixel(self, viewport_pixel: QPoint, factor: float) -> None:
        current = self.zoom_factor()
        target = min(self.MAX_ZOOM, max(self._fitted_zoom_factor(), current * float(factor)))
        if math.isclose(current, target, rel_tol=0.0, abs_tol=1e-9) or current <= 0:
            return
        applied = target / current
        scene_anchor = self.mapToScene(viewport_pixel)
        self.scale(applied, applied)
        mapped = self.mapFromScene(scene_anchor)
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(mapped.x()) - viewport_pixel.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(mapped.y()) - viewport_pixel.y())

    def _capture_zoom_snapshots(self) -> None:
        """Paint each visible tile once so later zoom frames only scale that image."""

        self._capturing_zoom_frame = True
        try:
            for tile in self._visible_tiles.values():
                bounds = tile.boundingRect()
                width = max(1, int(bounds.width()))
                height = max(1, int(bounds.height()))
                if width > 2048 or height > 2048:
                    tile._zoom_snapshot = None
                    continue
                snapshot = QPixmap(width, height)
                snapshot.fill(Qt.GlobalColor.transparent)
                painter = QPainter(snapshot)
                painter.translate(-bounds.left(), -bounds.top())
                option = QStyleOptionGraphicsItem()
                option.exposedRect = QRectF(bounds)
                tile.paint(painter, option, None)
                painter.end()
                tile._zoom_snapshot = snapshot
        finally:
            self._capturing_zoom_frame = False

    def _drop_zoom_snapshots(self) -> None:
        for tile in self._visible_tiles.values():
            tile._zoom_snapshot = None

    def _enter_zoom_render_mode(self) -> None:
        self._zoom_animating = True
        self._capture_zoom_snapshots()
        self.setRenderHints(self._zooming_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

    def _leave_zoom_render_mode(self) -> None:
        self._zoom_animating = False
        self._drop_zoom_snapshots()
        self.setRenderHints(self._steady_render_hints)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

    def _stop_zoom_animation(self, *, rebuild: bool) -> None:
        timer = getattr(self, "_zoom_animation_timer", None)
        if timer is not None:
            timer.stop()
        self._zoom_animation_anchor = None
        animating = getattr(self, "_zoom_animating", False)
        if animating:
            self._leave_zoom_render_mode()
        if rebuild and animating:
            self._after_transform()
        self._finish_zoom_profile("displayed" if rebuild else "interrupted")

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if not delta:
            super().wheelEvent(event)
            return
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._start_zoom_animation(event.position().toPoint(), wheel_zoom_factor(delta))
            event.accept()
            return
        scrollbar = (
            self.horizontalScrollBar()
            if modifiers & Qt.KeyboardModifier.ShiftModifier
            else self.verticalScrollBar()
        )
        pixel_delta = event.pixelDelta()
        step = pixel_delta.y() if not pixel_delta.isNull() else delta / 8.0
        scrollbar.setValue(scrollbar.value() - round(step * 3))
        event.accept()
