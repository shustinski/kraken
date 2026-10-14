"""test_widget_interactions: test middle button temporarily."""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from contour.graphics_view import (
    EditorTool,
)


class MiddleButtonTemporarilyMixin:
    def test_closing_view_finishes_active_zoom_profile(self) -> None:
        with patch(
            "contour.graphics.editor_view.scene_zoom_profiling_enabled",
            return_value=True,
        ):
            self.view._start_zoom_animation(QPoint(30, 30), 1.2)
        self.assertIsNotNone(self.view._scene_zoom_profile)

        self.view.close()

        self.assertIsNone(self.view._scene_zoom_profile)
        self.assertFalse(self.view._zoom_animation_timer.isActive())

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        self._app.processEvents()

    def test_middle_button_temporarily_hides_vectors_while_panning(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[..., 1] = 200
        self.view.set_gradient_overlay(overlay, opacity=0.6)
        overlay_item = self.view._editor_scene._gradient_overlay_item
        origin = self.view.mapFromScene(QPointF(50.0, 50.0))
        h_before = self.view.horizontalScrollBar().value()

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            origin,
        )
        self._app.processEvents()
        self.assertFalse(self.view._editor_scene.polygon_overlays_visible())
        self.assertTrue(overlay_item.isVisible())
        self.view.set_polygon_overlays_visible(True)
        self.assertFalse(self.view._editor_scene.polygon_overlays_visible())

        QTest.mouseMove(self.view.viewport(), origin + QPoint(30, -12), delay=10)
        self._app.processEvents()

        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            origin + QPoint(30, -12),
        )
        self._app.processEvents()

        self.assertTrue(self.view._editor_scene.polygon_overlays_visible())
        self.assertTrue(overlay_item.isVisible())
        self.assertEqual(len(self.view.get_polygons()), 1)
        self.assertEqual(self.view.current_tool, EditorTool.ADD_POLYGON)
        self.assertLessEqual(self.view.horizontalScrollBar().value(), h_before - 25)

    def test_f_key_fits_main_image_to_view(self) -> None:
        self.view.resetTransform()
        self.view.scale(4.0, 4.0)
        zoom_before = self.view.zoom_factor()

        QTest.keyClick(self.view, Qt.Key.Key_F)
        self._app.processEvents()

        self.assertLess(self.view.zoom_factor(), zoom_before)
        self.assertGreater(self.view.main_image_visible_fraction(), 0.99)

    def test_space_hold_hides_vectors_without_mutating_polygon_data(self) -> None:
        QTest.mouseClick(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(40, 40)
        )
        before = [(p.points[0], p.points[2]) for p in self.view.get_polygons()]
        QTest.keyPress(self.view, Qt.Key.Key_Space)

        self._app.processEvents()
        self.assertFalse(self.view._editor_scene.polygon_overlays_visible())
        after_press = [(p.points[0], p.points[2]) for p in self.view.get_polygons()]
        self.assertIsNotNone(self.view._editor_scene.selected_polygon_id())
        self.assertEqual(before, after_press)

        QTest.keyRelease(self.view, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertTrue(self.view._editor_scene.polygon_overlays_visible())
        after_release = [(p.points[0], p.points[2]) for p in self.view.get_polygons()]
        self.assertEqual(after_release, before)

    def test_space_hold_hides_recognition_mask_overlay(self) -> None:
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[..., 1] = 200
        self.view.set_gradient_overlay(overlay, opacity=0.6)
        overlay_item = self.view._editor_scene._gradient_overlay_item
        self.assertTrue(overlay_item.isVisible())

        QTest.keyPress(self.view, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertFalse(overlay_item.isVisible())
        self.assertFalse(overlay_item.pixmap().isNull())

        QTest.keyRelease(self.view, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertTrue(overlay_item.isVisible())

    def test_trace_pen_commits_polygonal_chain_on_enter(self) -> None:
        self.view.set_tool(EditorTool.TRACE_PEN)
        recorded: list[tuple[list[tuple[float, float]], float, bool]] = []

        def _record_trace(points: list[tuple[float, float]], width: float, erase: bool = False) -> bool:
            recorded.append((points, width, erase))
            self.view._editor_scene.cancel_pending_polygon()
            return True

        self.view._editor_scene.add_trace_stroke = _record_trace  # type: ignore[method-assign]

        for point in (QPointF(10.0, 10.0), QPointF(80.0, 10.0), QPointF(80.0, 70.0)):
            QTest.mouseClick(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                self.view.mapFromScene(point),
            )
        self._app.processEvents()

        self.assertEqual(len(self.view._editor_scene.pending_points_snapshot()), 3)

        QTest.keyClick(self.view, Qt.Key.Key_Return)
        self._app.processEvents()

        self.assertEqual(len(recorded), 1)
        points, width, erase = recorded[0]
        self.assertEqual(points, [(10.0, 10.0), (80.0, 10.0), (80.0, 70.0)])
        self.assertEqual(width, self.view._trace_width)
        self.assertFalse(erase)

    def test_trace_pen_allows_self_intersecting_chain(self) -> None:
        self.view.set_tool(EditorTool.TRACE_PEN)
        for point in (
            QPointF(10.0, 10.0),
            QPointF(80.0, 80.0),
            QPointF(80.0, 10.0),
            QPointF(10.0, 80.0),
        ):
            QTest.mouseClick(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                self.view.mapFromScene(point),
            )
        self._app.processEvents()
        self.assertEqual(
            self.view._editor_scene.pending_points_snapshot(),
            [(10.0, 10.0), (80.0, 80.0), (80.0, 10.0), (10.0, 80.0)],
        )

    def test_brush_and_trace_width_reject_one_pixel(self) -> None:
        self.view.set_brush_thickness(1.0)
        self.view.set_trace_width(1.0)
        self.assertGreaterEqual(self.view._brush_thickness, 2.0)
        self.assertGreaterEqual(self.view._trace_width, 2.0)

    def test_ctrl_wheel_keeps_scene_point_under_cursor_stable(self) -> None:
        self.view.fit_to_view()
        self._app.processEvents()
        pos = QPoint(90, 80)
        view_pos = self.view._viewport_to_view_point(pos)
        scene_before = self.view.mapToScene(view_pos)
        event = QWheelEvent(
            QPointF(pos),
            QPointF(pos),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        for _ in range(5):
            self._app.sendEvent(self.view.viewport(), event)
            self._app.processEvents()
        QTest.qWait(120)
        self._app.processEvents()
        scene_after = self.view.mapToScene(view_pos)
        self.assertAlmostEqual(scene_after.x(), scene_before.x(), delta=0.5)
        self.assertAlmostEqual(scene_after.y(), scene_before.y(), delta=0.5)
