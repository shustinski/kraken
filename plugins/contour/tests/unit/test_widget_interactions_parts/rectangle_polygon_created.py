"""test_widget_interactions: test rectangle polygon created."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.graphics_view import (
    EditorTool,
    PolygonCreateMode,
)


class RectanglePolygonCreatedMixin:
    def test_rectangle_polygon_is_created_with_two_clicks(self) -> None:
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)

        self._click_editor(Qt.MouseButton.LeftButton, QPointF(10.0, 12.0))
        self.assertEqual(self.view._drag_kind, "rect_polygon")
        self.assertEqual(len(self.view.get_polygons()), 0)

        self._move_editor_pointer(QPointF(40.0, 50.0))
        preview_bounds = self.view._editor_scene._preview_rect_item.path().boundingRect()
        self.assertGreater(preview_bounds.width(), 1.0)
        self.assertGreater(preview_bounds.height(), 1.0)

        self._click_editor(Qt.MouseButton.LeftButton, QPointF(40.0, 50.0))
        self.assertIsNone(self.view._drag_kind)
        polygons = self.view.get_polygons()
        self.assertEqual(len(polygons), 1)
        xs = [point[0] for point in polygons[0].points]
        ys = [point[1] for point in polygons[0].points]
        self.assertAlmostEqual(min(xs), 10.0, delta=2.0)
        self.assertAlmostEqual(min(ys), 12.0, delta=2.0)
        self.assertAlmostEqual(max(xs), 40.0, delta=2.0)
        self.assertAlmostEqual(max(ys), 50.0, delta=2.0)

    def test_rectangle_polygon_hold_and_release_does_not_finish(self) -> None:
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)
        start_pos = QPointF(10.0, 10.0)
        end_pos = QPointF(40.0, 40.0)

        self._press_editor(Qt.MouseButton.LeftButton, start_pos)
        self._move_editor_pointer(end_pos)
        self._release_editor(Qt.MouseButton.LeftButton, end_pos)
        self._app.processEvents()

        self.assertEqual(self.view._drag_kind, "rect_polygon")
        self.assertEqual(len(self.view.get_polygons()), 0)

        self._click_editor(Qt.MouseButton.LeftButton, QPointF(40.0, 40.0))
        self.assertEqual(len(self.view.get_polygons()), 1)
