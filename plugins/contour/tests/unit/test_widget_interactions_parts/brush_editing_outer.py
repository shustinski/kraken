"""test_widget_interactions: test brush editing outer."""
from __future__ import annotations
from PyQt6.QtCore import QPointF
from contour.application.vector_geometry_postprocess import VectorGeometrySettings


class BrushEditingOuterMixin:
    def test_brush_crossing_inner_contour_updates_hole_geometry(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        before_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]

        changed = self.view._editor_scene.add_brush_stroke(
            [(28.0, 50.0), (72.0, 50.0)],
            thickness=12.0,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        after_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertNotEqual(
            [(polygon.bbox, polygon.points) for polygon in before_holes],
            [(polygon.bbox, polygon.points) for polygon in after_holes],
        )

    def test_brush_editing_outer_ring_keeps_inner_object_if_not_touched(self) -> None:
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(56.0, 56.0))
        before_polygons = self.view.get_polygons()

        changed = self.view._editor_scene.add_brush_stroke(
            [(20.0, 28.0), (80.0, 28.0)],
            thickness=8.0,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        after_polygons = self.view.get_polygons()
        self.assertGreaterEqual(sum(1 for polygon in after_polygons if not polygon.is_hole), 2)
        self.assertTrue(
            any(not polygon.is_hole and polygon.bbox[0] >= 44 and polygon.bbox[1] >= 44 for polygon in after_polygons)
        )
        self.assertGreaterEqual(len(after_polygons), len(before_polygons))
