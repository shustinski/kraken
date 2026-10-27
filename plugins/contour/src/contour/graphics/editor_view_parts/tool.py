"""editor_view: tool."""
from __future__ import annotations
from time import perf_counter
from PyQt6.QtWidgets import QGraphicsView
from ...infrastructure.contact_placement_profiler import (
    MoveVertexToolActivationProfile,
)
from ...infrastructure.profiling import (
    move_vertex_tool_profiling_enabled,
)
from contour.graphics.tool_mode_logic import (
    normalize_editor_tool,
)
from contour.graphics.tools import EditorTool


class ToolMixin:
    def set_tool(self, tool: EditorTool) -> None:
        from contour.graphics.editor_view import write_profile_report

        tool = normalize_editor_tool(tool)
        if tool not in self._available_tools:
            if self._tool in self._available_tools:
                return
            tool = EditorTool.SELECT
        profile_activation = (
            tool == EditorTool.MOVE_VERTEX
            and move_vertex_tool_profiling_enabled()
        )
        profile_started_at: float | None = None
        if profile_activation:
            self._cancel_move_vertex_tool_profile_finish()
            if self._move_vertex_tool_profile is not None:
                self._finish_move_vertex_tool_profile("superseded")
            profile_started_at = perf_counter()
            polygons = self._editor_scene.get_polygons()
            self._move_vertex_tool_profile = MoveVertexToolActivationProfile.begin(
                polygon_count=len(polygons),
                vertex_count=sum(len(polygon.points) for polygon in polygons),
            )
            write_profile_report("[contour move-vertex-tool profiling] started")
        elif self._move_vertex_tool_profile is not None:
            self._cancel_move_vertex_tool_profile_finish()
            self._finish_move_vertex_tool_profile("superseded")
        self._editor_scene.cancel_polygon_edit_preview()
        self._drag_preview_points = None
        self._tool = tool
        self._select_press_polygon_id = None
        self._select_press_start = None
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag if tool == EditorTool.PAN else QGraphicsView.DragMode.NoDrag
        )
        if tool == EditorTool.ADD_POLYGON:
            self._editor_scene.set_pending_path_width(1.5, cosmetic=True)
        elif tool == EditorTool.BRUSH:
            self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        elif tool == EditorTool.TRACE_PEN:
            self._editor_scene.set_pending_path_width(self._trace_width, cosmetic=False)
        if tool not in (EditorTool.ADD_POLYGON, EditorTool.BRUSH, EditorTool.TRACE_PEN):
            self._editor_scene.cancel_pending_polygon()
            self._pending_polygon_erases = None
        if tool != EditorTool.ADD_POLYGON:
            self._cancel_rectangle_polygon()
        if tool != EditorTool.DELETE_VERTEX:
            self._editor_scene.clear_preview_rect()
        if tool != EditorTool.RULER:
            self._editor_scene.clear_measurement()
            self.rulerMeasurementChanged.emit("")
        if tool != EditorTool.ANTIALIAS:
            self._editor_scene.clear_vertex_preview()
        sync_phase_started_at = perf_counter()
        if profile_started_at is not None and self._move_vertex_tool_profile is not None:
            self._move_vertex_tool_profile.note_timing(
                "tool_setup",
                (sync_phase_started_at - profile_started_at) * 1000.0,
            )
        sync_refresh_ms = self._sync_all_editable_vertices_display()
        ui_phase_started_at = perf_counter()
        if profile_started_at is not None and self._move_vertex_tool_profile is not None:
            self._move_vertex_tool_profile.note_timing(
                "sync_vertices",
                sync_refresh_ms
                if sync_refresh_ms > 0.0
                else (ui_phase_started_at - sync_phase_started_at) * 1000.0,
            )
        self._update_tool_cursors()
        self._update_minimap_mouse_interaction()
        self.toolChanged.emit(tool)
        self._emit_effective_polygon_create_mode_changed()
        if profile_started_at is not None and self._move_vertex_tool_profile is not None:
            self._move_vertex_tool_profile.note_timing(
                "ui_finish",
                (perf_counter() - ui_phase_started_at) * 1000.0,
            )
            self._schedule_move_vertex_tool_profile_finish()
