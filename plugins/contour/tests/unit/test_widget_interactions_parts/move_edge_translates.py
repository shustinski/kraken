"""test_widget_interactions: test move edge translates."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from contour.application.vector_geometry_postprocess import VectorGeometrySettings
from contour.graphics_view import (
    EditorTool,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class MoveEdgeTranslatesMixin:
    def test_move_vertex_click_inside_selected_polygon_moves_nearest_vertex(self) -> None:
        poly = _rectangle_polygon(20, 20, 80, 80)
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=1.0, min_spike_interior_angle_deg=0.0)
        )
        self.view.set_polygons([poly])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        before = self.view.get_polygons()[0].points

        press_pos = self.view.mapFromScene(QPointF(50.0, 50.0))
        release_pos = self.view.mapFromScene(QPointF(62.0, 58.0))
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
        self.assertNotEqual(before, after)

    def test_move_edge_translates_both_endpoints(self) -> None:
        poly = _rectangle_polygon(20, 20, 80, 80)
        self.view.set_vector_geometry_settings(
            VectorGeometrySettings(min_outer_area_px2=1.0, min_spike_interior_angle_deg=0.0)
        )
        self.view.set_polygons([poly])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        before = self.view.get_polygons()[0].points

        press_pos = self.view.mapFromScene(QPointF(50.0, 20.0))
        release_pos = self.view.mapFromScene(QPointF(50.0, 35.0))
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
        self.assertNotEqual(before, after)
        self.assertAlmostEqual(after[0][1], 35.0, delta=1.0)
        self.assertAlmostEqual(after[1][1], 35.0, delta=1.0)
        self.assertEqual(after[2], before[2])
        self.assertEqual(after[3], before[3])

    def test_move_target_preview_highlights_edge_near_cursor(self) -> None:
        poly = _rectangle_polygon(20, 20, 80, 80)
        self.view.set_polygons([poly])
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        scene_pos = QPointF(50.0, 20.0)
        self.view._editor_scene.sync_move_target_preview(
            scene_pos,
            vertex_tolerance=2.0,
            edge_tolerance=8.0,
        )
        self.assertEqual(self.view._editor_scene._move_target_preview, ("edge", 1, 0))

    def test_move_target_preview_prefers_vertex_near_corner(self) -> None:
        poly = _rectangle_polygon(20, 20, 80, 80)
        self.view.set_polygons([poly])
        self.view.set_tool(EditorTool.MOVE_VERTEX)
        scene_pos = QPointF(22.0, 22.0)
        self.view._editor_scene.sync_move_target_preview(
            scene_pos,
            vertex_tolerance=8.0,
            edge_tolerance=8.0,
        )
        self.assertEqual(self.view._editor_scene._move_target_preview[0], "vertex")
