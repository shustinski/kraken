"""Tests for pure editor tool / polygon mode helpers."""

from __future__ import annotations

from contour.domain import PolygonData
from contour.graphics.tool_mode_logic import (
    EditorContentKind,
    available_editor_tools,
    editor_content_kind,
    effective_polygon_create_mode,
    normalize_editor_tool,
)
from contour.graphics.tools import EditorTool, PolygonCreateMode


def test_effective_polygon_mode_shift_flips_when_idle() -> None:
    assert (
        effective_polygon_create_mode(
            tool=EditorTool.ADD_POLYGON,
            base=PolygonCreateMode.POINTS,
            shift_held=True,
            has_pending_polygon=False,
        )
        == PolygonCreateMode.RECTANGLE
    )
    assert (
        effective_polygon_create_mode(
            tool=EditorTool.ADD_POLYGON,
            base=PolygonCreateMode.RECTANGLE,
            shift_held=True,
            has_pending_polygon=False,
        )
        == PolygonCreateMode.POINTS
    )


def test_effective_polygon_mode_pending_disables_shift_flip() -> None:
    assert (
        effective_polygon_create_mode(
            tool=EditorTool.ADD_POLYGON,
            base=PolygonCreateMode.POINTS,
            shift_held=True,
            has_pending_polygon=True,
        )
        == PolygonCreateMode.POINTS
    )


def test_effective_polygon_mode_non_polygon_tool_returns_base() -> None:
    assert (
        effective_polygon_create_mode(
            tool=EditorTool.BRUSH,
            base=PolygonCreateMode.RECTANGLE,
            shift_held=True,
            has_pending_polygon=False,
        )
        == PolygonCreateMode.RECTANGLE
    )


def test_normalize_legacy_select_area() -> None:
    assert normalize_editor_tool(EditorTool.SELECT_AREA) == EditorTool.SELECT
    assert normalize_editor_tool(EditorTool.SELECT) == EditorTool.SELECT


def test_available_tools_follow_current_content_kind() -> None:
    polygon = PolygonData(id=1, points=[(0, 0), (10, 0), (10, 10)])
    via = PolygonData(
        id=2,
        points=[(20, 20), (30, 20), (30, 30), (20, 30)],
        category="via",
        shape_hint="box",
    )

    assert editor_content_kind([]) == EditorContentKind.EMPTY
    assert EditorTool.ADD_POLYGON in available_editor_tools([])
    assert EditorTool.ADD_VIA in available_editor_tools([])

    assert editor_content_kind([polygon]) == EditorContentKind.POLYGONS
    assert EditorTool.ADD_POLYGON in available_editor_tools([polygon])
    assert EditorTool.ADD_VIA not in available_editor_tools([polygon])

    assert editor_content_kind([via]) == EditorContentKind.VIAS
    assert EditorTool.ADD_VIA in available_editor_tools([via])
    assert EditorTool.ADD_POLYGON not in available_editor_tools([via])
    assert EditorTool.MOVE_VERTEX not in available_editor_tools([via])

    assert editor_content_kind([polygon, via]) == EditorContentKind.MIXED
    assert available_editor_tools([polygon, via]) == {
        EditorTool.SELECT,
        EditorTool.PAN,
        EditorTool.RULER,
    }
