"""test_widget_interactions: test brush vertex spacing."""
from __future__ import annotations
from PyQt6.QtCore import QPointF
from contour.graphics_view import (
    EditorTool,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class BrushVertexSpacingMixin:
    def test_noop_brush_erase_does_not_create_undo_action(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        undo_before = self.view.undo_stack.count()

        changed = self.view._editor_scene.add_brush_stroke([(150.0, 150.0), (180.0, 180.0)], thickness=12.0, erase=True)

        self.assertFalse(changed)
        self.assertEqual(self.view.undo_stack.count(), undo_before)

    def test_brush_records_movement_of_at_least_one_pixel_as_segment(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        self.view._editor_scene.start_pending_polygon(for_brush=True)
        self.view._editor_scene.set_pending_path_width(12.0, cosmetic=False)

        self.view._editor_scene.append_brush_vertex(QPointF(40.0, 40.0), 12.0)
        self.view._editor_scene.append_brush_vertex(QPointF(41.2, 40.0), 12.0)

        points = self.view._editor_scene.pending_points_snapshot()
        self.assertGreaterEqual(len(points), 2)

    def test_brush_drops_vertices_closer_than_one_pixel(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        self.view._editor_scene.start_pending_polygon(for_brush=True)
        self.view._editor_scene.set_pending_path_width(12.0, cosmetic=False)
        self.view._append_brush_point(QPointF(40.0, 40.0))
        self.view._append_brush_point(QPointF(40.8, 40.0))

        points = self.view._editor_scene.pending_points_snapshot()
        self.assertEqual(len(points), 1)

    def test_brush_vertex_spacing_rule_is_one_image_pixel(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        self.view.resetTransform()
        self.view.scale(4.0, 4.0)
        self._app.processEvents()
        self.view._editor_scene.start_pending_polygon(for_brush=True)
        self.view._editor_scene.set_pending_path_width(12.0, cosmetic=False)
        self.view._append_brush_point(QPointF(40.0, 40.0))
        self.view._append_brush_point(QPointF(40.8, 40.0))
        points_after_small = self.view._editor_scene.pending_points_snapshot()
        self.assertEqual(len(points_after_small), 1)

        self.view._append_brush_point(QPointF(41.2, 40.0))
        points_after_large = self.view._editor_scene.pending_points_snapshot()
        self.assertGreaterEqual(len(points_after_large), 2)
