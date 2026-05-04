"""Pure helpers for editor tool / polygon-mode state (unit-tested)."""

from __future__ import annotations

from .tools import EditorTool, PolygonCreateMode


def effective_polygon_create_mode(
    *,
    tool: EditorTool,
    base: PolygonCreateMode,
    shift_held: bool,
    has_pending_polygon: bool,
) -> PolygonCreateMode:
    """Return the polygon draw mode used for new gestures.

    Shift temporarily flips points <-> rectangle when no stroke is in progress.
    While a point sequence is pending, the base mode is kept so Shift does not
    disrupt an in-progress polygon.
    """
    if tool != EditorTool.ADD_POLYGON:
        return base
    if has_pending_polygon:
        return base
    if shift_held:
        return PolygonCreateMode.RECTANGLE if base == PolygonCreateMode.POINTS else PolygonCreateMode.POINTS
    return base


def normalize_editor_tool(tool: EditorTool) -> EditorTool:
    """Merge legacy area-select into the unified select tool."""
    if tool == EditorTool.SELECT_AREA:
        return EditorTool.SELECT
    return tool
