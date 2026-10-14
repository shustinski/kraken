"""test_widget_interactions: test toolbar icons use."""
from __future__ import annotations
from PyQt6.QtGui import QRegion
from contour.graphics_view import (
    EditorTool,
)


class ToolbarIconsUseMixin:
    def test_checked_toolbar_tool_has_explicit_high_contrast_style(self) -> None:
        stylesheet = self.widget.styleSheet()

        self.assertIn("QToolButton:checked", stylesheet)
        self.assertIn("#16A34A", stylesheet)

        tool_button = self.widget._tool_buttons[EditorTool.SELECT]
        self.assertTrue(tool_button.isCheckable())
        self.assertTrue(tool_button.isChecked())
        self.assertIn("QToolButton:checked", tool_button.styleSheet())
        self.assertIn("#16A34A", tool_button.styleSheet())

    def test_toolbar_icons_use_the_complete_button_area(self) -> None:
        buttons = [
            *self.widget._tool_buttons.values(),
            self.widget.undo_button,
            self.widget.redo_button,
            self.widget.zoom_in_button,
            self.widget.zoom_out_button,
            self.widget.fit_button,
        ]

        for button in buttons:
            self.assertEqual(button.iconSize().width(), button.maximumWidth())
            self.assertEqual(button.iconSize().height(), button.maximumHeight())
            self.assertIn("padding: 0", button.styleSheet())
            self.assertTrue(button.toolTip())
            pixmap = button.icon().pixmap(button.iconSize())
            visible_rect = QRegion(pixmap.mask()).boundingRect()
            self.assertGreaterEqual(visible_rect.width(), pixmap.width() * 0.9)
            self.assertGreaterEqual(visible_rect.height(), pixmap.height() * 0.9)

    def test_hotkey_tool_switch_updates_toolbar_checked_state(self) -> None:
        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()

        self.assertTrue(self.widget._tool_buttons[EditorTool.BRUSH].isChecked())
        self.assertFalse(self.widget._tool_buttons[EditorTool.SELECT].isChecked())

        self.widget.polygon_editor.set_tool(EditorTool.SELECT)
        self._app.processEvents()

        self.assertTrue(self.widget._tool_buttons[EditorTool.SELECT].isChecked())
        self.assertFalse(self.widget._tool_buttons[EditorTool.BRUSH].isChecked())

    def test_delete_polygon_tool_is_not_exposed_in_toolbar(self) -> None:
        self.assertNotIn(EditorTool.DELETE_POLYGON, self.widget._tool_buttons)

    def test_antialias_tool_is_exposed_in_toolbar(self) -> None:
        self.assertIn(EditorTool.ANTIALIAS, self.widget._tool_buttons)

    def test_trace_pen_tool_is_exposed_in_toolbar(self) -> None:
        self.assertIn(EditorTool.TRACE_PEN, self.widget._tool_buttons)

    def test_antialias_settings_are_only_visible_for_antialias_tool(self) -> None:
        self.assertTrue(self.widget._antialias_toolbar_block.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.ANTIALIAS)
        self._app.processEvents()

        self.assertFalse(self.widget._antialias_toolbar_block.isHidden())
        self.assertFalse(self.widget.antialias_grade_label.isHidden())
        self.assertFalse(self.widget.antialias_grade_spin.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()

        self.assertTrue(self.widget._antialias_toolbar_block.isHidden())

    def test_trace_pen_settings_are_only_visible_for_trace_pen_tool(self) -> None:
        self.assertTrue(self.widget._trace_toolbar_block.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.TRACE_PEN)
        self._app.processEvents()

        self.assertFalse(self.widget._trace_toolbar_block.isHidden())
        self.assertFalse(self.widget.trace_width_label.isHidden())
        self.assertFalse(self.widget.trace_width_spin.isHidden())

        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()

        self.assertTrue(self.widget._trace_toolbar_block.isHidden())

    def test_tool_settings_strip_stays_in_fixed_toolbar_position(self) -> None:
        strip_index = self.widget._editor_toolbar_layout.indexOf(self.widget.tool_settings_strip)

        self.widget.polygon_editor.set_tool(EditorTool.ADD_POLYGON)
        self._app.processEvents()
        polygon_index = self.widget._editor_toolbar_layout.indexOf(self.widget.tool_settings_strip)

        self.widget.polygon_editor.set_tool(EditorTool.BRUSH)
        self._app.processEvents()
        brush_index = self.widget._editor_toolbar_layout.indexOf(self.widget.tool_settings_strip)

        self.assertEqual(strip_index, polygon_index)
        self.assertEqual(strip_index, brush_index)
        self.assertFalse(self.widget._brush_toolbar_block.isHidden())
        self.assertTrue(self.widget._polygon_toolbar_block.isHidden())
