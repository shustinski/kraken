"""test_widget_interactions: test middle pan during."""
from __future__ import annotations
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from contour.graphics_view import (
    EditorTool,
)


class MiddlePanDuringMixin:
    def test_middle_pan_during_brush_does_not_shift_brush_position(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.BRUSH)
        self.view.set_brush_thickness(12.0)
        start = self.view.mapFromScene(QPointF(30.0, 30.0))

        QTest.mousePress(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mousePress(self.view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(self.view.viewport(), start + QPoint(90, 0), delay=10)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(90, 0),
        )
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(90, 0),
        )
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(polygons)
        # Stroke remains around the original press location; pan does not drag the brush.
        left, top, width, height = polygons[0].bbox
        self.assertLessEqual(left, 31)
        self.assertLessEqual(top, 31)
        self.assertGreaterEqual(left + width, 29)
        self.assertGreaterEqual(top + height, 29)

    def test_middle_pan_during_brush_at_image_edge_keeps_brush_anchor(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.BRUSH)
        self.view.set_brush_thickness(12.0)
        self.view.resetTransform()
        self.view.scale(4.0, 4.0)
        self._app.processEvents()

        edge_scene = QPointF(95.0, 50.0)
        self.view.centerOn(edge_scene)
        self._app.processEvents()
        start = self.view.mapFromScene(edge_scene)

        QTest.mousePress(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mousePress(self.view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, start)
        # Push pan strongly so viewport hits scroll limits at image boundary.
        for _ in range(3):
            QTest.mouseMove(self.view.viewport(), start + QPoint(-600, 0), delay=8)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(-600, 0),
        )
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start + QPoint(-600, 0),
        )
        self._app.processEvents()

        polygons = self.view.get_polygons()
        self.assertTrue(polygons)
        left, top, width, height = polygons[0].bbox
        center_x = left + width / 2.0
        center_y = top + height / 2.0
        self.assertAlmostEqual(center_x, edge_scene.x(), delta=2.0)
        self.assertAlmostEqual(center_y, edge_scene.y(), delta=2.0)
