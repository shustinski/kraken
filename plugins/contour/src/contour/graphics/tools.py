"""Editor tool enums used by the scene/view classes and the UI."""

from __future__ import annotations

from enum import StrEnum


class EditorTool(StrEnum):
    SELECT = "select"
    SELECT_AREA = "select_area"
    PAN = "pan"
    RULER = "ruler"
    ADD_POLYGON = "add_polygon"
    BRUSH = "brush"
    TRACE_PEN = "trace_pen"
    ADD_VIA = "add_via"
    ADD_VERTEX = "add_vertex"
    DELETE_VERTEX = "delete_vertex"
    MOVE_VERTEX = "move_vertex"
    ANTIALIAS = "antialias"
    DELETE_POLYGON = "delete_polygon"


class PolygonCreateMode(StrEnum):
    POINTS = "points"
    RECTANGLE = "rectangle"


class BrushMode(StrEnum):
    FREEFORM = "freeform"
    ANGLED = "angled"


class DeleteVertexMode(StrEnum):
    SINGLE = "single"
    AREA = "area"
