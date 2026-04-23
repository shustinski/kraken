"""Backward-compatibility shim.

The implementation has moved to ``polygon_widget.graphics``. This module is
kept so existing imports (``from polygon_widget.graphics_view import ...``)
continue to work; new code should import from the package instead.
"""

from __future__ import annotations

from .graphics import (
    BrushMode,
    DeleteVertexMode,
    EditorTool,
    PolygonCreateMode,
    PolygonEditorScene,
    PolygonEditorView,
)

__all__ = [
    "BrushMode",
    "DeleteVertexMode",
    "EditorTool",
    "PolygonCreateMode",
    "PolygonEditorScene",
    "PolygonEditorView",
]
