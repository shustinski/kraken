"""test_widget_interactions: test small independent erase."""
from __future__ import annotations
from unittest.mock import patch
from PyQt6.QtCore import QPointF
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics_view import (
    BrushMode,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class SmallIndependentEraseMixin:
    def test_angled_brush_pan_preserves_endpoint_drawn_before_pan(self) -> None:
        self.view.set_brush_mode(BrushMode.ANGLED)
        self.view._start_brush_drag(QPointF(30.0, 30.0), erase=False)
        self.view._last_pointer_scene_pos = QPointF(60.0, 30.0)
        self.view._brush_pan_guard = True

        with patch.object(self.view._editor_scene, "add_brush_stroke") as commit:
            self.view._commit_brush_drag(QPointF(-100.0, 30.0))

        commit.assert_called_once_with(
            [(30.0, 30.0), (60.0, 30.0)], self.view._brush_thickness, erase=False
        )
        self.assertFalse(self.view._brush_pan_guard)

    def test_closed_brush_contour_preserves_empty_center(self) -> None:
        self.view.set_polygons([])
        points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]

        changed = self.view._editor_scene.add_brush_stroke(points, thickness=10.0)
        self._app.processEvents()

        self.assertTrue(changed)
        polygons = self.view.get_polygons()
        self.assertTrue(any(polygon.is_hole for polygon in polygons))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertFalse(outer_item.contains(QPointF(50.0, 50.0)))

    def test_closed_brush_contour_fills_center_below_manual_min_hole_area(self) -> None:
        self.view.set_polygons([])
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=10.0, min_hole_area_to_remove_px2=150.0)
        )
        points = [
            (30.0, 30.0),
            (50.0, 30.0),
            (50.0, 50.0),
            (30.0, 50.0),
            (30.0, 30.0),
        ]

        changed = self.view._editor_scene.add_brush_stroke(points, thickness=10.0)
        self._app.processEvents()

        self.assertTrue(changed)
        polygons = self.view.get_polygons()
        self.assertFalse(any(polygon.is_hole for polygon in polygons))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertTrue(outer_item.contains(QPointF(40.0, 40.0)))

    def test_brush_erase_fills_hole_below_manual_min_hole_area(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=10.0, min_hole_area_to_remove_px2=11.0)
        )

        changed = self.view._editor_scene.add_brush_stroke(
            [(44.0, 44.0), (46.0, 44.0), (46.0, 46.0), (44.0, 46.0), (44.0, 44.0)],
            thickness=1.0,
            erase=True,
        )
        self._app.processEvents()

        self.assertFalse(changed)
        self.assertEqual(len(self.view.get_polygons()), 1)
        self.assertFalse(any(polygon.is_hole for polygon in self.view.get_polygons()))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertTrue(outer_item.contains(QPointF(45.0, 45.0)))

    def test_small_independent_erase_is_rejected_when_existing_hole_is_above_min_area(self) -> None:
        outer = _rectangle_polygon(0, 0, 100, 100)
        outer = PolygonData(id=1, points=outer.points, area=outer.area, perimeter=outer.perimeter, bbox=outer.bbox)
        hole_points = [(40.0, 40.0), (54.0, 40.0), (54.0, 54.0), (40.0, 54.0)]
        hole_area, hole_perimeter, hole_bbox = compute_polygon_metrics(hole_points)
        hole = PolygonData(
            id=2,
            points=hole_points,
            is_hole=True,
            parent_id=1,
            area=hole_area,
            perimeter=hole_perimeter,
            bbox=hole_bbox,
        )
        self.view.set_polygons([outer, hole])
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=10.0, min_hole_area_to_remove_px2=100.0)
        )
        before = [(polygon.id, polygon.points, polygon.is_hole) for polygon in self.view.get_polygons()]

        changed = self.view._editor_scene.add_rectangle_polygon(
            QPointF(10.0, 10.0),
            QPointF(13.0, 13.0),
            erase=True,
        )
        self._app.processEvents()

        self.assertFalse(changed)
        after = [(polygon.id, polygon.points, polygon.is_hole) for polygon in self.view.get_polygons()]
        self.assertEqual(after, before)

    def test_brush_erase_keeps_hole_above_manual_min_hole_area(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=10.0, min_hole_area_to_remove_px2=11.0)
        )

        changed = self.view._editor_scene.add_brush_stroke(
            [(45.0, 45.0)],
            thickness=10.0,
            erase=True,
        )
        self._app.processEvents()

        self.assertTrue(changed)
        self.assertTrue(any(polygon.is_hole for polygon in self.view.get_polygons()))
        outer_item = next(item for item in self.view._editor_scene._polygon_items.values() if not item.polygon.is_hole)
        self.assertFalse(outer_item.contains(QPointF(45.0, 45.0)))

    def test_small_rectangle_fully_inside_existing_contour_is_not_drawn(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view._editor_scene.set_vector_geometry_settings(VectorGeometrySettings(min_hole_area_to_remove_px2=11.0))

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(46.0, 46.0))
        self._app.processEvents()

        self.assertFalse(added)
        self.assertEqual(len(self.view.get_polygons()), 1)

    def test_small_rectangle_fully_inside_existing_cutout_is_not_drawn(self) -> None:
        self.view.set_polygons([])
        self.view._editor_scene.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=10.0, min_hole_area_to_remove_px2=11.0)
        )
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)
        before = [(polygon.id, polygon.points, polygon.is_hole) for polygon in self.view.get_polygons()]

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(46.0, 46.0))
        self._app.processEvents()

        self.assertFalse(added)
        after = [(polygon.id, polygon.points, polygon.is_hole) for polygon in self.view.get_polygons()]
        self.assertEqual(after, before)

    def test_can_draw_polygon_inside_existing_cutout(self) -> None:
        self.view.set_polygons([])
        ring_points = [
            (30.0, 30.0),
            (70.0, 30.0),
            (70.0, 70.0),
            (30.0, 70.0),
            (30.0, 30.0),
        ]
        self.view._editor_scene.add_brush_stroke(ring_points, thickness=10.0)

        added = self.view._editor_scene.add_rectangle_polygon(QPointF(44.0, 44.0), QPointF(56.0, 56.0))
        self._app.processEvents()

        self.assertTrue(added)
        polygons = self.view.get_polygons()
        self.assertEqual(sum(1 for polygon in polygons if polygon.is_hole), 1)
        self.assertGreaterEqual(sum(1 for polygon in polygons if not polygon.is_hole), 2)
        self.assertTrue(
            any(not polygon.is_hole and polygon.bbox[0] >= 44 and polygon.bbox[1] >= 44 for polygon in polygons)
        )
