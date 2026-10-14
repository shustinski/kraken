"""test_widget_interactions: test add vertex click."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from contour.graphics_view import (
    EditorTool,
)

from test_widget_interactions_parts.oversampled_rectangle_polygon import _rectangle_polygon


class AddVertexClickMixin:
    def test_add_vertex_click_on_unselected_polygon_selects_and_edits_it(self) -> None:
        first = _rectangle_polygon(10, 10, 40, 40)
        second = _rectangle_polygon(60, 10, 90, 40)
        second.id = 2
        self.view.set_polygons([first, second])
        self.view._editor_scene.select_polygon(1)
        self.view.set_tool(EditorTool.ADD_VERTEX)

        select_pos = self.view.mapFromScene(QPointF(75.0, 25.0))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            select_pos,
        )
        self._app.processEvents()

        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertEqual(len(polygons[2].points), 4)
        self.assertEqual(len(polygons[1].points), 4)
        self.assertEqual(self.view._editor_scene.selected_polygon_id(), 2)

        # A point exactly on the edge is redundant and intentionally collapsed.
        add_pos = self.view.mapFromScene(QPointF(75.0, 5.0))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            add_pos,
        )
        self._app.processEvents()

        polygons = {polygon.id: polygon for polygon in self.view.get_polygons()}
        self.assertEqual(len(polygons[2].points), 5)
        self.assertEqual(len(polygons[1].points), 4)
        self.assertEqual(self.view._editor_scene.selected_polygon_id(), 2)
