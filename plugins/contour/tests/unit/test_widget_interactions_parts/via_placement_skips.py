"""test_widget_interactions: test via placement skips."""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
from PyQt6.QtCore import QPointF
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.graphics_view import (
    PolygonEditorScene,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _app,
    _rectangle_polygon,
)


class ViaPlacementSkipsMixin:
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def test_brush_preview_keeps_configured_width_across_mouse_updates(self) -> None:
        scene = PolygonEditorScene()
        scene.start_pending_polygon(for_brush=True)
        scene.set_pending_path_width(40.0, cosmetic=False)
        with patch.object(scene, "_tool_preview_path") as exact_preview:
            scene.append_brush_vertex(QPointF(10.0, 10.0), 40.0)
            scene.update_pending_cursor(QPointF(110.0, 10.0))
            first_height = scene._pending_path_item.boundingRect().height()
            scene.update_pending_cursor(QPointF(120.0, 10.0))
            second_height = scene._pending_path_item.boundingRect().height()

        exact_preview.assert_not_called()
        self.assertGreater(first_height, 35.0)
        self.assertGreater(second_height, 35.0)

    def test_polygon_cursor_preview_defers_exact_commit_validation(self) -> None:
        scene = PolygonEditorScene()
        for point in (QPointF(10.0, 10.0), QPointF(80.0, 10.0), QPointF(80.0, 80.0)):
            scene.append_pending_point(point)

        with patch("contour.graphics.editor_scene.polygon_commit_acceptability") as acceptability:
            scene.update_pending_cursor(QPointF(20.0, 70.0))

        acceptability.assert_not_called()
        self.assertFalse(scene._pending_path_item.path().isEmpty())

    def test_local_polygon_add_does_not_run_full_frame_postprocess(self) -> None:
        scene = PolygonEditorScene()
        scene.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                min_spike_interior_angle_deg=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        polygons = []
        for index in range(100):
            polygon = _rectangle_polygon(index * 20, 0, index * 20 + 10, 10)
            polygon.id = index + 1
            polygons.append(polygon)
        scene.set_polygons(polygons)

        with patch("contour.graphics.editor_scene.postprocess_after_editor_mutation") as full_cleanup:
            changed = scene.add_rectangle_polygon(QPointF(2200.0, 0.0), QPointF(2210.0, 10.0))

        self.assertTrue(changed)
        full_cleanup.assert_not_called()

    def test_hover_hit_testing_passes_only_spatial_candidates(self) -> None:
        scene = PolygonEditorScene()
        polygons = []
        for index in range(100):
            polygon = _rectangle_polygon(index * 300, 0, index * 300 + 20, 20)
            polygon.id = index + 1
            polygons.append(polygon)
        scene.set_polygons(polygons)

        with patch(
            "contour.graphics.editor_scene.resolve_hover_polygon_id",
            return_value=None,
        ) as resolve_hover:
            scene.sync_conductor_hover_highlight(QPointF(10.0, 10.0))

        candidate_polygons = resolve_hover.call_args.args[0]
        self.assertEqual(set(candidate_polygons), {1})

    def test_append_brush_vertex_does_not_densify_long_jump(self) -> None:
        scene = PolygonEditorScene()
        scene.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        scene.start_pending_polygon(for_brush=True)
        scene.append_brush_vertex(QPointF(10.0, 10.0), 40.0)
        scene.append_brush_vertex(QPointF(310.0, 10.0), 40.0)

        pending = scene.pending_points_snapshot()
        self.assertEqual(pending, [(10, 10), (310, 10)])

        changed = scene.add_brush_stroke(pending, thickness=20.0)
        polygons = scene.get_polygons()
        self.assertTrue(changed)
        self.assertTrue(polygons)
        bbox = polygons[0].bbox
        self.assertLessEqual(bbox[0], 150)
        self.assertGreaterEqual(bbox[0] + bbox[2], 150)

    def test_trace_pen_adds_fixed_width_conductor_polygon(self) -> None:
        scene = PolygonEditorScene()

        changed = scene.add_trace_stroke([(10.0, 20.0), (110.0, 20.0)], width=16.0, erase=False)
        polygons = scene.get_polygons()

        self.assertTrue(changed)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(polygons[0].category, "conductor")
        self.assertEqual(polygons[0].shape_hint, "trace_pen")
        self.assertGreater(polygons[0].area, 1400.0)

    def test_trace_pen_clips_new_polygon_to_image_bounds(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((40, 80), dtype=np.uint8))

        changed = scene.add_trace_stroke([(-20.0, 20.0), (100.0, 20.0)], width=16.0, erase=False)
        polygons = scene.get_polygons()

        self.assertTrue(changed)
        self.assertEqual(len(polygons), 1)
        for x_coord, y_coord in polygons[0].points:
            self.assertGreaterEqual(x_coord, 0)
            self.assertLessEqual(x_coord, 80)
            self.assertGreaterEqual(y_coord, 0)
            self.assertLessEqual(y_coord, 40)

    def test_via_placement_rejects_overlapping_existing_via(self) -> None:
        scene = PolygonEditorScene()

        self.assertTrue(scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0))
        self.assertFalse(scene.add_via_at(QPointF(24.0, 20.0), 10.0, 10.0))
        self.assertTrue(scene.add_via_at(QPointF(40.0, 20.0), 10.0, 10.0))

        self.assertEqual(len(scene.get_polygons()), 2)

    def test_via_placement_rejects_rect_outside_image_bounds(self) -> None:
        scene = PolygonEditorScene()
        scene.set_image(np.zeros((40, 40), dtype=np.uint8))

        self.assertFalse(scene.add_via_at(QPointF(2.0, 20.0), 10.0, 10.0))
        self.assertTrue(scene.add_via_at(QPointF(20.0, 20.0), 10.0, 10.0))

        self.assertEqual(len(scene.get_polygons()), 1)

    def test_via_placement_skips_global_cleanup_and_refreshes_only_new_selection(self) -> None:
        scene = PolygonEditorScene()
        contacts = []
        for index in range(100):
            contact = _rectangle_polygon(index * 15, 0, index * 15 + 8, 8)
            contact.id = index + 1
            contact.category = "via"
            contact.shape_hint = "box"
            contacts.append(contact)
        scene.set_polygons(contacts)
        scene.set_image(np.zeros((60, 1600), dtype=np.uint8))

        with (
            patch.object(scene, "_maybe_push_vector_postprocess") as postprocess,
            patch.object(
                scene,
                "_refresh_polygon_selection_item",
                wraps=scene._refresh_polygon_selection_item,
            ) as refresh_item,
        ):
            added = scene.add_via_at(QPointF(20.0, 30.0), 8.0, 8.0)

        self.assertTrue(added)
        postprocess.assert_not_called()
        self.assertEqual(refresh_item.call_count, 1)
        self.assertEqual(scene.selected_polygon_id(), 101)
