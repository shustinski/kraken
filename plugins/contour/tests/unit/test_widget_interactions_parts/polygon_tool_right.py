"""test_widget_interactions: test polygon tool right."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.graphics_view import (
    EditorTool,
    PolygonCreateMode,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _net_outline_area


class PolygonToolRightMixin:
    def test_polygon_tool_right_click_on_existing_starts_erase_not_delete(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)
        before = [polygon.clone() for polygon in self.view.get_polygons()]

        self._click_editor(Qt.MouseButton.RightButton, QPointF(50.0, 50.0))

        self.assertEqual(self.view._drag_kind, "rect_polygon")
        self.assertTrue(self.view._drag_erases)
        after = self.view.get_polygons()
        self.assertEqual(len(after), len(before))
        self.assertEqual(after[0].points, before[0].points)

    def test_polygon_tool_right_click_inside_existing_erases_area(self) -> None:
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)
        initial_area = _net_outline_area(self.view.get_polygons())

        self._click_editor(Qt.MouseButton.RightButton, QPointF(40.0, 40.0))
        self._move_editor_pointer(QPointF(60.0, 60.0))
        self._click_editor(Qt.MouseButton.RightButton, QPointF(60.0, 60.0))

        polygons = self.view.get_polygons()
        final_area = _net_outline_area(polygons)
        self.assertLess(final_area, initial_area)
        self.assertTrue(polygons)
        self.assertTrue(any(polygon.is_hole for polygon in polygons))

    def test_polygon_tool_right_click_while_drawing_keeps_erase_drawing_behavior(self) -> None:
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.POINTS)

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(5.0, 5.0)),
        )
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(10.0, 5.0)),
        )
        self._app.processEvents()

        self.assertEqual(len(self.view.get_polygons()), 1)
        self.assertEqual(len(self.view._editor_scene.pending_points_snapshot()), 2)
