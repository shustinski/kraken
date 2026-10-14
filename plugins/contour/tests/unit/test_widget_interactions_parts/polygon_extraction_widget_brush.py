"""test_widget_interactions: polygon extraction widget brush."""
from __future__ import annotations
import unittest
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
)
from contour.graphics_view import (
    BrushMode,
    DeleteVertexMode,
    EditorTool,
    PolygonCreateMode,
)
from contour.widget import PolygonExtractionWidget

from test_widget_interactions_parts.oversampled_rectangle_polygon import (
    _app,
    _rectangle_polygon,
)

class PolygonExtractionWidgetBrushModeUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def setUp(self) -> None:
        self.widget = PolygonExtractionWidget()

    def tearDown(self) -> None:
        self.widget.close()
        self.widget.deleteLater()
        self._app.processEvents()

    def test_brush_mode_combo_exposes_all_brush_modes(self) -> None:
        modes = [
            str(self.widget.brush_mode_combo.itemData(index)) for index in range(self.widget.brush_mode_combo.count())
        ]
        self.assertEqual(self.widget.brush_mode_combo.count(), 2)
        self.assertEqual(modes, ["freeform", "angled"])

    def test_trace_pen_width_control_updates_editor(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.TRACE_PEN)
        self.widget.trace_width_spin.setValue(24)
        self._app.processEvents()

        self.assertEqual(self.widget.polygon_editor._trace_width, 24.0)

    def test_contact_tool_uses_plain_width_and_height_labels(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.ADD_VIA)
        self._app.processEvents()

        self.assertEqual(self.widget.via_width_label.text(), "Ширина")
        self.assertEqual(self.widget.via_height_label.text(), "Высота")

        self.widget.via_width_spin.setValue(13)
        self.widget.via_height_spin.setValue(21)
        self._app.processEvents()
        self.assertEqual(self.widget.polygon_editor._via_width, 13.0)
        self.assertEqual(self.widget.polygon_editor._via_height, 21.0)

    def test_polygon_mode_indicator_is_hidden_but_mode_switch_stays_operational(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self.widget.polygon_mode_combo.setCurrentIndex(
            self.widget.polygon_mode_combo.findData(PolygonCreateMode.RECTANGLE)
        )
        self._app.processEvents()

        self.assertFalse(self.widget.polygon_draw_mode_indicator.isVisible())
        self.assertEqual(self.widget.polygon_editor.effective_polygon_create_mode(), PolygonCreateMode.RECTANGLE)

    def test_space_hold_hides_vectors_after_selecting_drawing_tool_button(self) -> None:
        self.widget.polygon_editor.set_image(np.zeros((100, 100), dtype=np.uint8))
        self.widget.polygon_editor.set_polygons([_rectangle_polygon(20, 20, 80, 80)])
        self.widget.show()
        self._app.processEvents()

        QTest.mouseClick(self.widget._tool_buttons[EditorTool.BRUSH], Qt.MouseButton.LeftButton)
        self._app.processEvents()

        self.assertEqual(QApplication.focusWidget(), self.widget.polygon_editor)
        QTest.keyPress(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertFalse(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

        QTest.keyRelease(self.widget.polygon_editor, Qt.Key.Key_Space)
        self._app.processEvents()
        self.assertTrue(self.widget.polygon_editor._editor_scene.polygon_overlays_visible())

    def test_tool_mode_cycle_matches_shift_click_targets(self) -> None:
        self.widget._cycle_editor_tool_mode(EditorTool.ADD_POLYGON)
        self.widget._cycle_editor_tool_mode(EditorTool.BRUSH)
        self.widget._cycle_editor_tool_mode(EditorTool.DELETE_VERTEX)
        self._app.processEvents()

        self.assertEqual(self.widget.polygon_mode_combo.currentData(), PolygonCreateMode.POINTS)
        self.assertEqual(self.widget.brush_mode_combo.currentData(), BrushMode.FREEFORM)
        self.assertEqual(self.widget.delete_vertex_mode_combo.currentData(), DeleteVertexMode.SINGLE)

    def test_editor_tool_modes_default_to_rectangle_angled_and_area(self) -> None:
        self.assertEqual(self.widget.polygon_mode_combo.currentData(), PolygonCreateMode.RECTANGLE)
        self.assertEqual(self.widget.brush_mode_combo.currentData(), BrushMode.ANGLED)
        self.assertEqual(self.widget.delete_vertex_mode_combo.currentData(), DeleteVertexMode.AREA)
        self.assertEqual(self.widget.polygon_editor.effective_polygon_create_mode(), PolygonCreateMode.RECTANGLE)

    def test_shift_key_cycles_active_tool_mode_and_updates_combo(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        QTest.keyClick(self.widget.polygon_editor, Qt.Key.Key_Shift)
        QTest.keyClick(self.widget.polygon_editor, Qt.Key.Key_Shift)
        self._app.processEvents()

        self.assertEqual(self.widget.brush_mode_combo.currentData(), BrushMode.ANGLED)

    def test_toolbar_hides_incompatible_tools_and_falls_back_to_select(self) -> None:
        via = _rectangle_polygon(10, 10, 20, 20)
        via.category = "via"
        via.shape_hint = "box"
        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self.widget.polygon_editor.set_polygons([via])
        self._app.processEvents()

        self.assertEqual(self.widget.polygon_editor.current_tool, EditorTool.SELECT)
        self.assertFalse(self.widget._tool_buttons[EditorTool.ADD_VIA].isHidden())
        for tool in (
            EditorTool.ADD_POLYGON,
            EditorTool.BRUSH,
            EditorTool.TRACE_PEN,
            EditorTool.ADD_VERTEX,
            EditorTool.DELETE_VERTEX,
            EditorTool.MOVE_VERTEX,
            EditorTool.ANTIALIAS,
        ):
            self.assertTrue(self.widget._tool_buttons[tool].isHidden())
        self.widget.polygon_editor.set_tool(EditorTool.ADD_VIA)
        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self.assertEqual(self.widget.polygon_editor.current_tool, EditorTool.ADD_VIA)

        self.widget.polygon_editor.set_polygons([_rectangle_polygon(0, 0, 30, 30)])
        self._app.processEvents()
        self.assertTrue(self.widget._tool_buttons[EditorTool.ADD_VIA].isHidden())
        self.assertFalse(self.widget._tool_buttons[EditorTool.ADD_POLYGON].isHidden())
