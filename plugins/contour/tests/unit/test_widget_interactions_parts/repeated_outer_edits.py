"""test_widget_interactions: test repeated outer edits."""
from __future__ import annotations
from contour.application.vector_geometry_postprocess import VectorGeometrySettings


class RepeatedOuterEditsMixin:
    def test_repeated_outer_edits_do_not_expand_untouched_inner_contour(self) -> None:
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
        initial_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(initial_holes), 1)

        for y_coord in (28.0, 26.0, 24.0):
            changed = self.view._editor_scene.add_brush_stroke(
                [(20.0, y_coord), (80.0, y_coord)],
                thickness=4.0,
            )
            self.assertTrue(changed)

        final_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(final_holes), 1)
        self.assertEqual(final_holes[0].bbox, initial_holes[0].bbox)
        self.assertEqual(final_holes[0].points, initial_holes[0].points)

    def test_repeated_inner_edge_edits_do_not_expand_hole_bbox(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (90.0, 30.0),
            (90.0, 90.0),
            (30.0, 90.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        initial_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(initial_holes), 1)
        initial_bbox = initial_holes[0].bbox

        for _index in range(4):
            changed = self.view._editor_scene.add_brush_stroke(
                [(20.0, 34.0), (100.0, 34.0)],
                thickness=2.0,
            )
            self.assertTrue(changed)

        final_holes = [polygon.clone() for polygon in self.view.get_polygons() if polygon.is_hole]
        self.assertEqual(len(final_holes), 1)
        self.assertGreaterEqual(final_holes[0].bbox[0], initial_bbox[0])
        self.assertGreaterEqual(final_holes[0].bbox[1], initial_bbox[1])
        self.assertLessEqual(final_holes[0].bbox[2], initial_bbox[2])
        self.assertLessEqual(final_holes[0].bbox[3], initial_bbox[3])
