"""Animated wheel zoom for a QGraphicsView.

A wheel notch sets a target scale; the view glides there in ``DURATION_MS`` with an ease-out,
about 60 frames per second. Fast notches add up to one target. The scene point under the
cursor stays under the cursor. High-resolution wheels and touchpads zoom in proportion to
the delta. ``stepped`` fires on every frame, ``settled`` once the view stopped for
``SETTLE_MS``: heavy work (loading, full-quality images) belongs there.
"""

from __future__ import annotations

import math
import time

from PyQt6.QtCore import QObject, QPointF, QTimer, pyqtSignal
from PyQt6.QtWidgets import QGraphicsView

DURATION_MS = 150
FRAME_MS = 16
SETTLE_MS = 100
WHEEL_NOTCH = 120.0


def wheel_zoom_factor(event, step: float) -> float:
    """``step`` per wheel notch; a smaller delta (touchpad, fine wheel) zooms proportionally less."""

    delta = float(event.angleDelta().y())
    if delta == 0.0:
        delta = float(event.pixelDelta().y()) * 4.0
    if delta == 0.0:
        return 1.0
    return float(step) ** (delta / WHEEL_NOTCH)


class SmoothZoom(QObject):
    stepped = pyqtSignal()
    settled = pyqtSignal()

    def __init__(self, view: QGraphicsView, *, min_scale: float, max_scale: float) -> None:
        super().__init__(view)
        self._view = view
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._start_scale = 1.0
        self._target_scale = 1.0
        self._started = 0.0
        self._anchor_view = QPointF()
        self._anchor_scene = QPointF()
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(SETTLE_MS)
        self._settle_timer.timeout.connect(self.settled.emit)

    @property
    def animating(self) -> bool:
        return self._timer.isActive()

    @property
    def target_scale(self) -> float:
        return self._target_scale if self.animating else self._current_scale()

    def _current_scale(self) -> float:
        return abs(float(self._view.transform().m11())) or 1.0

    def zoom_by(self, factor: float, anchor_viewport_pos: QPointF) -> bool:
        """Start (or extend) a glide; False when the limit is already reached."""

        current = self._current_scale()
        target = min(self._max_scale, max(self._min_scale, self.target_scale * float(factor)))
        if math.isclose(target, current, rel_tol=1e-4) and not self.animating:
            return False
        self._start_scale = current
        self._target_scale = target
        self._started = time.perf_counter()
        self._anchor_view = QPointF(anchor_viewport_pos)
        self._anchor_scene = self._view.mapToScene(anchor_viewport_pos.toPoint())
        self._settle_timer.stop()
        if not self._timer.isActive():
            self._timer.start()
        self._tick()
        return True

    def stop(self) -> None:
        self._timer.stop()
        self._settle_timer.stop()

    def _tick(self) -> None:
        progress = min(1.0, (time.perf_counter() - self._started) * 1000.0 / DURATION_MS)
        eased = 1.0 - (1.0 - progress) ** 3
        scale = self._start_scale * (self._target_scale / self._start_scale) ** eased
        self._apply(scale)
        self.stepped.emit()
        if progress >= 1.0:
            self._timer.stop()
            self._settle_timer.start()

    def _apply(self, scale: float) -> None:
        view = self._view
        current = self._current_scale()
        if current <= 0.0:
            return
        previous_anchor = view.transformationAnchor()
        view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        view.scale(scale / current, scale / current)
        view.setTransformationAnchor(previous_anchor)
        # Keep the scene point that was under the cursor under the cursor.
        drift = view.viewportTransform().map(self._anchor_scene) - self._anchor_view
        horizontal = view.horizontalScrollBar()
        vertical = view.verticalScrollBar()
        horizontal.setValue(horizontal.value() + round(drift.x()))
        vertical.setValue(vertical.value() + round(drift.y()))
