"""Animated wheel zoom: glides to the target, keeps the point under the cursor, adds up notches."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from karakal.ui.smooth_zoom import SmoothZoom, wheel_zoom_factor


def _wheel(view: QGraphicsView, position: QPointF, delta: int, *, ctrl: bool = False) -> QWheelEvent:
    return QWheelEvent(
        position,
        QPointF(view.viewport().mapToGlobal(position.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _view(qtbot) -> QGraphicsView:
    view = QGraphicsView()
    scene = QGraphicsScene(view)
    scene.setSceneRect(QRectF(0, 0, 4000, 4000))
    view.setScene(scene)
    view.resize(600, 400)
    qtbot.addWidget(view)
    view.show()
    return view


def test_wheel_factor_is_proportional_to_the_delta(qtbot) -> None:
    view = _view(qtbot)
    center = QPointF(300, 200)
    assert math.isclose(wheel_zoom_factor(_wheel(view, center, 120), 1.2), 1.2)
    assert math.isclose(wheel_zoom_factor(_wheel(view, center, -120), 1.2), 1 / 1.2)
    assert math.isclose(wheel_zoom_factor(_wheel(view, center, 60), 1.2), 1.2**0.5)


def test_glides_to_the_sum_of_notches_and_keeps_the_point_under_the_cursor(qtbot) -> None:
    view = _view(qtbot)
    zoom = SmoothZoom(view, min_scale=0.05, max_scale=64.0)
    anchor = QPointF(150, 120)
    under_cursor = view.viewportTransform().inverted()[0].map(anchor)
    zoom.zoom_by(1.5, anchor)
    zoom.zoom_by(1.5, anchor)
    assert zoom.animating
    assert view.transform().m11() < 2.25
    with qtbot.waitSignal(zoom.settled, timeout=2000):
        pass
    assert math.isclose(view.transform().m11(), 2.25, rel_tol=1e-3)
    drift = view.viewportTransform().map(under_cursor) - anchor
    # Scroll bars move by whole pixels: up to a pixel of rounding is invisible.
    assert abs(drift.x()) <= 2.0 and abs(drift.y()) <= 2.0


def test_limits_hold(qtbot) -> None:
    view = _view(qtbot)
    zoom = SmoothZoom(view, min_scale=0.5, max_scale=2.0)
    zoom.zoom_by(100.0, QPointF(10, 10))
    with qtbot.waitSignal(zoom.settled, timeout=2000):
        pass
    assert math.isclose(view.transform().m11(), 2.0, rel_tol=1e-3)
    assert zoom.zoom_by(1.5, QPointF(10, 10)) is False


def test_matrix_ctrl_wheel_zooms_smoothly(qtbot) -> None:
    from karakal.core.domain import FrameRecord
    from karakal.ui.matrix_view import MatrixLayoutConfig, MatrixListWidget

    view = MatrixListWidget()
    qtbot.addWidget(view)
    view.resize(800, 600)
    view.show()
    view.set_layout_config(MatrixLayoutConfig(mode="indexed_grid", frames_per_row=50))
    view.set_records([FrameRecord(f"F_{index:04d}.png", f"F_{index:04d}.png") for index in range(2000)])
    start = view.transform().m11()
    view.wheelEvent(_wheel(view, QPointF(400, 300), 120, ctrl=True))
    assert view._smooth_zoom.animating
    with qtbot.waitSignal(view._smooth_zoom.settled, timeout=2000):
        pass
    assert math.isclose(view.transform().m11(), start * 1.12, rel_tol=1e-3)


def test_frame_window_shows_crisp_pixels_from_2x(qtbot) -> None:
    from karakal.ui.details_dialog import _OverlayGraphicsView

    view = _OverlayGraphicsView()
    qtbot.addWidget(view)
    view.resize(400, 400)
    view.show()
    qtbot.waitExposed(view)
    image = QImage(200, 200, QImage.Format.Format_RGB32)
    image.fill(0x808080)
    item = QGraphicsPixmapItem(QPixmap.fromImage(image))
    item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
    view.scene().addItem(item)
    view.resetTransform()
    qtbot.waitUntil(lambda: item.transformationMode() == Qt.TransformationMode.SmoothTransformation, timeout=2000)
    view.scale(3.0, 3.0)
    view.viewport().update()
    qtbot.waitUntil(lambda: item.transformationMode() == Qt.TransformationMode.FastTransformation, timeout=2000)
