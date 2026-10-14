"""test_widget_interactions: test vertex edit tools."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from contour.graphics_view import (
    DeleteVertexMode,
    EditorTool,
    PolygonCreateMode,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _oversampled_rectangle_polygon,
    _rectangle_polygon,
)


class VertexEditToolsMixin:
    def test_escape_cancels_in_progress_rectangle_polygon(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)

        self._click_editor(Qt.MouseButton.LeftButton, QPointF(10.0, 10.0))
        self._move_editor_pointer(QPointF(40.0, 40.0))
        QTest.keyClick(self.view, Qt.Key.Key_Escape)
        self._app.processEvents()

        self.assertIsNone(self.view._drag_kind)
        self.assertTrue(self.view._editor_scene._preview_rect_item.path().isEmpty())
        self.assertEqual(self.view.get_polygons(), [])

    def test_right_click_cancels_in_progress_left_rectangle_polygon(self) -> None:
        self.view.set_polygons([])
        self.view.set_tool(EditorTool.ADD_POLYGON)
        self.view.set_polygon_create_mode(PolygonCreateMode.RECTANGLE)

        self._click_editor(Qt.MouseButton.LeftButton, QPointF(10.0, 10.0))
        self._move_editor_pointer(QPointF(40.0, 40.0))
        self._click_editor(Qt.MouseButton.RightButton, QPointF(40.0, 40.0))

        self.assertIsNone(self.view._drag_kind)
        self.assertEqual(self.view.get_polygons(), [])

    def test_brush_tool_hover_syncs_conductor_highlight_on_mouse_move(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view.set_tool(EditorTool.BRUSH)
        self._move_editor_pointer(QPointF(50.0, 50.0))
        self.assertEqual(self.view._editor_scene._hover_conductor_polygon_id, 1)

    def test_conductor_hover_fills_polygon_instead_of_outline_only(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view.set_tool(EditorTool.SELECT)
        self._move_editor_pointer(QPointF(50.0, 50.0))
        item = self.view._editor_scene._polygon_items[1]
        self.assertGreater(item.brush().color().alpha(), 64)
        self.assertTrue(item._edge_highlight_item.path().isEmpty())

    def test_hole_hover_highlights_inner_polygon_not_parent(self) -> None:
        outer = _rectangle_polygon(0, 0, 80, 80)
        outer.id = 1
        hole = _rectangle_polygon(20, 20, 40, 40)
        hole.id = 2
        hole.is_hole = True
        hole.parent_id = 1
        self.view.set_polygons([outer, hole])
        self.view.set_tool(EditorTool.SELECT)
        self._move_editor_pointer(QPointF(30.0, 30.0))
        self.assertEqual(self.view._editor_scene._hover_conductor_polygon_id, 2)

    def test_nested_polygon_hover_highlights_smaller_inner_polygon(self) -> None:
        outer = _rectangle_polygon(0, 0, 120, 120)
        outer.id = 1
        inner = _rectangle_polygon(40, 40, 90, 90)
        inner.id = 2
        self.view.set_polygons([outer, inner])
        self.view.set_tool(EditorTool.SELECT)
        self._move_editor_pointer(QPointF(65.0, 65.0))
        self.assertEqual(self.view._editor_scene._hover_conductor_polygon_id, 2)

    def test_trace_pen_tool_hover_syncs_conductor_highlight_on_mouse_move(self) -> None:
        self.view.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.view.set_tool(EditorTool.TRACE_PEN)
        self._move_editor_pointer(QPointF(50.0, 50.0))
        self.assertEqual(self.view._editor_scene._hover_conductor_polygon_id, 1)

    def test_brush_drag_skips_conductor_hover_sync_on_mouse_move(self) -> None:
        self.view.set_tool(EditorTool.BRUSH)
        calls: list[QPointF] = []
        original_sync = self.view._editor_scene.sync_conductor_hover_highlight
        self.view._editor_scene.sync_conductor_hover_highlight = lambda pos: calls.append(pos)  # type: ignore[method-assign]
        try:
            start_pos = self.view.mapFromScene(QPointF(25.0, 25.0))
            move_pos = self.view.mapFromScene(QPointF(75.0, 75.0))
            QTest.mousePress(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                start_pos,
            )
            QTest.mouseMove(self.view.viewport(), move_pos)
            QTest.mouseRelease(
                self.view.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                move_pos,
            )
            self._app.processEvents()
        finally:
            self.view._editor_scene.sync_conductor_hover_highlight = original_sync  # type: ignore[method-assign]
        self.assertEqual(calls, [])

    def test_antialias_tool_hover_shows_vertices_and_click_simplifies_polygon(self) -> None:
        self.view.set_polygons([_oversampled_rectangle_polygon(20, 20, 80, 80)])
        self.view.set_tool(EditorTool.ANTIALIAS)
        self.view.set_antialias_grade(1)
        hover_pos = self.view.mapFromScene(QPointF(50.0, 50.0))

        QTest.mouseMove(self.view.viewport(), hover_pos)
        self._app.processEvents()

        item = self.view._editor_scene._polygon_items[1]
        self.assertGreater(len(item._handles), 0)
        before_count = len(self.view.get_polygons()[0].points)

        QTest.mouseClick(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, hover_pos)
        self._app.processEvents()

        after = self.view.get_polygons()[0]
        self.assertLess(len(after.points), before_count)
        self.assertEqual(self.view.current_tool, EditorTool.ANTIALIAS)

    def test_antialias_click_on_outer_also_smooths_hole(self) -> None:
        outer = _oversampled_rectangle_polygon(10, 10, 90, 90)
        hole = _oversampled_rectangle_polygon(30, 30, 70, 70)
        hole.id = 2
        hole.is_hole = True
        hole.parent_id = 1
        self.view.set_polygons([outer, hole])
        self.view.set_tool(EditorTool.ANTIALIAS)
        self.view.set_antialias_grade(2)
        before = {polygon.id: len(polygon.points) for polygon in self.view.get_polygons()}

        # Click on metal between outer and hole so hit-test returns the outer id.
        click_pos = self.view.mapFromScene(QPointF(20.0, 50.0))
        QTest.mouseClick(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, click_pos)
        self._app.processEvents()

        after = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertLess(len(after[1].points), before[1])
        self.assertLess(len(after[2].points), before[2])
        self.assertTrue(after[2].is_hole)
        self.assertEqual(after[2].parent_id, 1)

    def test_vertex_edit_tools_show_all_polygon_vertices(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(50, 50, 80, 80)
        second.id = 2
        self.view.set_polygons([first, second])
        self.view.set_tool(EditorTool.SELECT)
        self.assertEqual(len(self.view._editor_scene._polygon_items[1]._handles), 0)
        self.assertEqual(len(self.view._editor_scene._polygon_items[2]._handles), 0)

        self.view.set_tool(EditorTool.MOVE_VERTEX)
        self.assertEqual(len(self.view._editor_scene._polygon_items[1]._handles), 4)
        self.assertEqual(len(self.view._editor_scene._polygon_items[2]._handles), 4)

        self.view.set_tool(EditorTool.ANTIALIAS)
        self.assertEqual(len(self.view._editor_scene._polygon_items[1]._handles), 4)
        self.assertEqual(len(self.view._editor_scene._polygon_items[2]._handles), 4)

        self.view.set_delete_vertex_mode(DeleteVertexMode.AREA)
        self.view.set_tool(EditorTool.DELETE_VERTEX)
        self.assertEqual(len(self.view._editor_scene._polygon_items[1]._handles), 0)

        self.view.set_delete_vertex_mode(DeleteVertexMode.SINGLE)
        self.assertEqual(len(self.view._editor_scene._polygon_items[1]._handles), 4)
        self.assertEqual(len(self.view._editor_scene._polygon_items[2]._handles), 4)

        self.view.set_tool(EditorTool.SELECT)
        self.assertEqual(len(self.view._editor_scene._polygon_items[1]._handles), 0)

    def test_antialias_tool_drag_simplifies_polygons_in_area(self) -> None:
        first = _oversampled_rectangle_polygon(20, 20, 40, 40)
        second = _oversampled_rectangle_polygon(60, 60, 80, 80)
        second.id = 2
        self.view.set_polygons([first, second])
        self.view.set_tool(EditorTool.ANTIALIAS)
        self.view.set_antialias_grade(1)
        before_counts = {polygon.id: len(polygon.points) for polygon in self.view.get_polygons()}

        start_pos = self.view.mapFromScene(QPointF(15.0, 15.0))
        end_pos = self.view.mapFromScene(QPointF(85.0, 85.0))
        QTest.mousePress(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start_pos)
        QTest.mouseMove(self.view.viewport(), end_pos)
        QTest.mouseRelease(self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end_pos)
        self._app.processEvents()

        after_counts = {polygon.id: len(polygon.points) for polygon in self.view.get_polygons()}
        self.assertLess(after_counts[1], before_counts[1])
        self.assertLess(after_counts[2], before_counts[2])
        self.assertEqual(self.view.current_tool, EditorTool.ANTIALIAS)
