"""test_widget_interactions: test delete vertices area."""
from __future__ import annotations
from unittest.mock import patch
from PyQt6.QtCore import QPointF, QRectF

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class DeleteVerticesAreaMixin:
    def test_delete_vertices_area_affects_unselected_polygons_too(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])
        self.view._editor_scene.select_polygon(1)

        deleted = self.view._editor_scene.delete_vertices_in_rect(QRectF(QPointF(56.0, 6.0), QPointF(66.0, 16.0)))

        self.assertEqual(deleted, 1)
        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertEqual(len(polygons[2].points), 3)
        self.assertEqual(len(polygons[1].points), 4)

    def test_delete_vertices_area_uses_incremental_commit_without_global_union(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])
        scene = self.view._editor_scene
        before = {polygon.id: polygon.points for polygon in self.view.get_polygons()}

        with (
            patch(
                "contour.graphics.editor_scene.union_after_removing_polygon_ids",
                side_effect=AssertionError("point-only edits must not run a global union"),
            ),
            patch.object(scene, "_bulk_restore_polygons", wraps=scene._bulk_restore_polygons) as bulk_restore,
        ):
            deleted = scene.delete_vertices_in_rect(QRectF(QPointF(56.0, 6.0), QPointF(66.0, 16.0)))
            self.assertEqual(deleted, 1)
            bulk_restore.assert_not_called()

            scene.undo_stack.undo()
            self.assertEqual({polygon.id: polygon.points for polygon in self.view.get_polygons()}, before)
            bulk_restore.assert_not_called()

            scene.undo_stack.redo()
            polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
            self.assertEqual(len(polygons[2].points), 3)
            bulk_restore.assert_not_called()

    def test_delete_vertices_area_removes_vector_when_ring_cannot_survive(self) -> None:
        triangle = _rectangle_polygon(10, 10, 40, 40)
        triangle.points = [(10.0, 10.0), (40.0, 10.0), (25.0, 40.0)]
        rectangle = _rectangle_polygon(60, 10, 90, 40)
        rectangle.id = 2
        self.view.set_polygons([triangle, rectangle])

        deleted = self.view._editor_scene.delete_vertices_in_rect(QRectF(QPointF(5.0, 5.0), QPointF(45.0, 15.0)))

        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertGreaterEqual(deleted, 2)
        self.assertNotIn(1, polygons)
        self.assertIn(2, polygons)
        self.assertEqual(len(polygons[2].points), 4)

    def test_delete_vertices_area_deletes_polygon_when_all_vertices_are_inside(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])

        deleted = self.view._editor_scene.delete_vertices_in_rect(QRectF(QPointF(5.0, 5.0), QPointF(45.0, 45.0)))

        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertEqual(deleted, 4)
        self.assertNotIn(1, polygons)
        self.assertEqual(len(polygons[2].points), 4)

    def test_delete_vertices_area_collapses_axis_aligned_extra_points(self) -> None:
        first = _rectangle_polygon(0, 0, 10, 10)
        first.points = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])

        deleted = self.view._editor_scene.delete_vertices_in_rect(QRectF(QPointF(8.0, 8.0), QPointF(12.0, 12.0)))

        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertGreaterEqual(deleted, 1)
        self.assertEqual(polygons[1].points, [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])
        self.assertEqual(len(polygons[2].points), 4)

    def test_delete_vertices_area_preview_highlights_all_touched_polygons(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])
        scene = self.view._editor_scene

        with (
            patch.object(scene, "_refresh_all_items", wraps=scene._refresh_all_items) as refresh_all,
            patch.object(
                scene,
                "_refresh_polygon_items_by_id",
                wraps=scene._refresh_polygon_items_by_id,
            ) as refresh_changed,
        ):
            scene.preview_delete_vertices_in_rect(QPointF(5.0, 5.0), QPointF(65.0, 15.0))

            self.assertEqual(scene._delete_area_highlight_ids, {1, 2})
            refresh_all.assert_not_called()
            self.assertEqual(refresh_changed.call_count, 1)
            self.assertEqual(set(refresh_changed.call_args.args), {1, 2})

            refresh_changed.reset_mock()
            scene.clear_preview_rect()
            self.assertEqual(scene._delete_area_highlight_ids, set())
            refresh_all.assert_not_called()
            self.assertEqual(refresh_changed.call_count, 1)
            self.assertEqual(set(refresh_changed.call_args.args), {1, 2})
