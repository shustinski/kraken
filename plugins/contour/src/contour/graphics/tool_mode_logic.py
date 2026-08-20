"""Pure helpers for editor tool / polygon-mode state (unit-tested)."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from ..domain import PolygonData
from .tools import EditorTool, PolygonCreateMode


class EditorContentKind(StrEnum):
    EMPTY = "empty"
    POLYGONS = "polygons"
    VIAS = "vias"
    MIXED = "mixed"


NAVIGATION_TOOLS = frozenset(
    {
        EditorTool.SELECT,
        EditorTool.PAN,
        EditorTool.RULER,
    }
)
POLYGON_TOOLS = frozenset(
    {
        EditorTool.ADD_POLYGON,
        EditorTool.BRUSH,
        EditorTool.TRACE_PEN,
        EditorTool.ADD_VERTEX,
        EditorTool.DELETE_VERTEX,
        EditorTool.MOVE_VERTEX,
        EditorTool.ANTIALIAS,
        EditorTool.DELETE_POLYGON,
    }
)


def is_via_polygon(polygon: PolygonData) -> bool:
    return polygon.category == "via" or polygon.shape_hint == "box"


def is_recognized_via(polygon: PolygonData) -> bool:
    return is_via_polygon(polygon) and polygon.recognition_score is not None


def editor_content_kind(polygons: Iterable[PolygonData]) -> EditorContentKind:
    has_vias = False
    has_polygons = False
    for polygon in polygons:
        if is_via_polygon(polygon):
            has_vias = True
        else:
            has_polygons = True
        if has_vias and has_polygons:
            return EditorContentKind.MIXED
    if has_vias:
        return EditorContentKind.VIAS
    if has_polygons:
        return EditorContentKind.POLYGONS
    return EditorContentKind.EMPTY


def available_editor_tools(polygons: Iterable[PolygonData]) -> frozenset[EditorTool]:
    kind = editor_content_kind(polygons)
    if kind == EditorContentKind.EMPTY:
        return frozenset(EditorTool) - {EditorTool.SELECT_AREA}
    if kind == EditorContentKind.POLYGONS:
        return NAVIGATION_TOOLS | POLYGON_TOOLS
    if kind == EditorContentKind.VIAS:
        return NAVIGATION_TOOLS | {EditorTool.ADD_VIA}
    return NAVIGATION_TOOLS


def apply_conductor_recognition_tool_lock(
    tools: frozenset[EditorTool],
    *,
    enabled: bool,
) -> frozenset[EditorTool]:
    """Keep only inspect/navigate tools while conductor recognition is active."""
    if not enabled:
        return tools
    return frozenset(tools) & NAVIGATION_TOOLS


def can_add_polygon(polygons: Iterable[PolygonData]) -> bool:
    return editor_content_kind(polygons) in {EditorContentKind.EMPTY, EditorContentKind.POLYGONS}


def can_add_via(polygons: Iterable[PolygonData]) -> bool:
    return editor_content_kind(polygons) in {EditorContentKind.EMPTY, EditorContentKind.VIAS}


def can_add_polygon_set(existing: Iterable[PolygonData], added: Iterable[PolygonData]) -> bool:
    combined_kind = editor_content_kind([*existing, *added])
    return combined_kind != EditorContentKind.MIXED


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
