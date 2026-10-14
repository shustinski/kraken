"""test_widget_interactions: test move vertex keeps."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.domain import PolygonData, compute_polygon_metrics
from contour.graphics_view import (
    EditorTool,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class MoveVertexKeepsMixin:
    def test_move_vertex_keeps_closed_duplicate_endpoint_together(self) -> None:
        points = [
            (20.0, 20.0),
            (80.0, 20.0),
            (80.0, 80.0),
            (20.0, 80.0),
            (20.0, 20.0),
        ]
        area, perimeter, bbox = compute_polygon_metrics(points)
        poly = PolygonData(id=1, points=points, area=area, perimeter=perimeter, bbox=bbox)
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(
                min_outer_area_px2=1.0,
                min_hole_area_to_remove_px2=0.0,
                drop_three_vertex_triangle_artifacts=False,
            )
        )
        self.view.set_polygons([poly])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)

        press_pos = self.view.mapFromScene(QPointF(20.0, 20.0))
        release_pos = self.view.mapFromScene(QPointF(30.0, 30.0))
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            press_pos,
        )
        QTest.mouseMove(self.view.viewport(), release_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            release_pos,
        )
        self._app.processEvents()

        after = self.view.get_polygons()[0].points
        self.assertAlmostEqual(after[0][0], 30.0, places=1)
        self.assertAlmostEqual(after[0][1], 30.0, places=1)
        self.assertEqual(after[-1], after[0])

    def test_move_vertex_merges_overlapping_conductors(self) -> None:
        first = _rectangle_polygon(20, 20, 50, 50)
        second = _rectangle_polygon(55, 20, 85, 50)
        second.id = 2
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=1.0, min_spike_interior_angle_deg=0.0)
        )
        self.view.set_polygons([first, second])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        before_undo_count = self.view.undo_stack.count()

        press_pos = self.view.mapFromScene(QPointF(50.0, 20.0))
        release_pos = self.view.mapFromScene(QPointF(65.0, 20.0))
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            press_pos,
        )
        QTest.mouseMove(self.view.viewport(), release_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            release_pos,
        )
        self._app.processEvents()

        roots = [polygon for polygon in self.view.get_polygons() if polygon.parent_id is None and not polygon.is_hole]
        self.assertEqual(len(roots), 1)
        self.assertEqual(self.view.undo_stack.count(), before_undo_count + 1)
