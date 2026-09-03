"""Editor tool enums used by the scene/view classes and the UI."""

from __future__ import annotations

from enum import StrEnum

from ..infrastructure.runtime_config import config_float

# Brush and trace-pen width in image pixels. 1 px is too thin to author reliably.
MIN_MANUAL_STROKE_WIDTH_PX = config_float("editor", "min_manual_stroke_width_px", 2.0, minimum=0.1)


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
