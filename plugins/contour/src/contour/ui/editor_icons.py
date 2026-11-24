"""Editor toolbar/tool icons.

The artwork lives in the monochrome SVG set loaded by :mod:`contour.ui.icon_set`;
this module maps editor tools and actions to icon names and re-applies the
themed icons and button styles after a theme change.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QAbstractButton, QToolButton

from ..graphics.tools import EditorTool
from . import icon_set
from .styles import toolbar_button_stylesheet

TOOLBAR_ICON_SIZE_PX = 24
TOOLBAR_BUTTON_SIZE_PX = 34
TOOLBAR_ICON_CANVAS_SIZE_PX = icon_set.ICON_GRID_PX

_TOOL_ICON_NAMES: dict[EditorTool, str] = {
    EditorTool.SELECT: "select",
    EditorTool.SELECT_AREA: "select_area",
    EditorTool.PAN: "pan",
    EditorTool.RULER: "ruler",
    EditorTool.ADD_POLYGON: "add_polygon",
    EditorTool.BRUSH: "brush",
    EditorTool.TRACE_PEN: "trace_pen",
    EditorTool.ADD_VIA: "add_via",
    EditorTool.ADD_VERTEX: "add_vertex",
    EditorTool.DELETE_VERTEX: "delete_vertex",
    EditorTool.MOVE_VERTEX: "move_vertex",
    EditorTool.ANTIALIAS: "antialias",
    EditorTool.DELETE_POLYGON: "delete_polygon",
}

_ACTION_ICON_NAMES: dict[str, str] = {
    "undo": "undo",
    "redo": "redo",
    "zoom_in": "zoom_in",
    "zoom_out": "zoom_out",
    "fit": "fit",
    "antialias": "antialias",
    "antialias_all": "antialias_all",
}

# Widget attribute -> action icon, for buttons rebuilt on a theme change.
_ACTION_BUTTONS: dict[str, str] = {
    "undo_button": "undo",
    "redo_button": "redo",
    "zoom_in_button": "zoom_in",
    "zoom_out_button": "zoom_out",
    "fit_button": "fit",
    "antialias_opened_cif_button": "antialias_all",
}


def create_editor_tool_icon(tool: EditorTool) -> QIcon:
    return icon_set.icon(_TOOL_ICON_NAMES[tool])


def create_editor_action_icon(action: str) -> QIcon:
    return icon_set.icon(_ACTION_ICON_NAMES.get(action, "fit"))


def refresh_editor_icons(widget: Any) -> None:
    """Rebuild toolbar icons and button styles in the current theme's colours."""

    colors = icon_set.icon_colors()
    tool_buttons: dict[EditorTool, QAbstractButton] = getattr(widget, "_tool_buttons", None) or {}
    for tool, button in tool_buttons.items():
        button.setIcon(icon_set.icon(_TOOL_ICON_NAMES[tool], colors))
        if isinstance(button, QToolButton):
            button.setStyleSheet(toolbar_button_stylesheet(colors, checkable=button.isCheckable()))
    for attribute, action in _ACTION_BUTTONS.items():
        action_button = getattr(widget, attribute, None)
        if not isinstance(action_button, QAbstractButton):
            continue
        action_button.setIcon(icon_set.icon(_ACTION_ICON_NAMES[action], colors))
        if isinstance(action_button, QToolButton):
            action_button.setStyleSheet(toolbar_button_stylesheet(colors, checkable=action_button.isCheckable()))


__all__ = [
    "TOOLBAR_BUTTON_SIZE_PX",
    "TOOLBAR_ICON_CANVAS_SIZE_PX",
    "TOOLBAR_ICON_SIZE_PX",
    "create_editor_action_icon",
    "create_editor_tool_icon",
    "refresh_editor_icons",
]
