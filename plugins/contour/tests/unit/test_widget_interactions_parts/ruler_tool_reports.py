"""test_widget_interactions: test ruler tool reports."""
from __future__ import annotations
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from contour.graphics_view import (
    EditorTool,
    PolygonCreateMode,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _net_outline_area


class RulerToolReportsMixin:
    def test_ruler_tool_reports_measurement_without_changing_polygons(self) -> None:
        measurements: list[str] = []
        self.view.rulerMeasurementChanged.connect(measurements.append)
        self.view.set_tool(EditorTool.RULER)

        start_pos = self.view.mapFromScene(QPointF(20.0, 20.0))
        end_pos = self.view.mapFromScene(QPointF(80.0, 60.0))

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start_pos,
        )
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end_pos,
        )
        self._app.processEvents()

        self.assertTrue(measurements)
        self.assertIn("L=", measurements[-1])
        self.assertFalse(self.view._editor_scene._measurement_item.path().isEmpty())
        self.assertTrue(self.view._editor_scene._measurement_label_item.isVisible())
        self.assertIn("L=", self.view._editor_scene._measurement_label_item.text())
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_ruler_shift_snaps_measurement_to_45_degree_step(self) -> None:
        self.view.set_tool(EditorTool.RULER)
        start = QPointF(20.0, 20.0)
        target = QPointF(80.0, 50.0)

        snapped = self.view._ruler_target(start, target, Qt.KeyboardModifier.ShiftModifier)

        dx = snapped.x() - start.x()
        dy = snapped.y() - start.y()
        self.assertAlmostEqual(abs(dx), abs(dy), delta=1e-6)

    def test_right_button_brush_erases_existing_polygon_area(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        initial_area = _net_outline_area(self.view.get_polygons())
        start_pos = self.view.mapFromScene(QPointF(50.0, 20.0))
        end_pos = self.view.mapFromScene(QPointF(50.0, 80.0))

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            start_pos,
        )
        QTest.mouseMove(self.view.viewport(), end_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            end_pos,
        )
        self._app.processEvents()

        final_area = _net_outline_area(self.view.get_polygons())
        self.assertLess(final_area, initial_area)

    def _press_editor(self, button: Qt.MouseButton, scene_pos: QPointF) -> None:
        pos = QPointF(self.view.mapFromScene(scene_pos))
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            pos,
            button,
            button,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.mousePressEvent(event)

    def _release_editor(self, button: Qt.MouseButton, scene_pos: QPointF) -> None:
        pos = QPointF(self.view.mapFromScene(scene_pos))
        event = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            pos,
            button,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.mouseReleaseEvent(event)

    def _click_editor(self, button: Qt.MouseButton, scene_pos: QPointF) -> None:
        self._press_editor(button, scene_pos)
        self._release_editor(button, scene_pos)
        self._app.processEvents()

    def _move_editor_pointer(self, scene_pos: QPointF) -> None:
        viewport_pos = QPointF(self.view.mapFromScene(scene_pos))
        move_event = QMouseEvent(
            QEvent.Type.MouseMove,
            viewport_pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.mouseMoveEvent(move_event)
        self._app.processEvents()

    def test_right_button_rectangle_polygon_erases_existing_polygon_area(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)
        initial_area = _net_outline_area(self.view.get_polygons())

        self._click_editor(Qt.MouseButton.RightButton, QPointF(5.0, 5.0))
        self._move_editor_pointer(QPointF(50.0, 50.0))
        self._click_editor(Qt.MouseButton.RightButton, QPointF(50.0, 50.0))

        final_area = _net_outline_area(self.view.get_polygons())
        self.assertLess(final_area, initial_area)
