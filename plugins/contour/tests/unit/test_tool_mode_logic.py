"""Tests for pure editor tool / polygon mode helpers."""

from __future__ import annotations

from contour.graphics.tool_mode_logic import effective_polygon_create_mode, normalize_editor_tool
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
