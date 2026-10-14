"""editor_view: cycle active tool mode."""
from __future__ import annotations
import cProfile
import io
import pstats
from math import hypot
from time import perf_counter
from PyQt6.QtCore import (
    QPointF,
    Qt,
    QTimer,
)
from ...infrastructure.contact_placement_profiler import (
    ContactDragProfile,
)
from ...infrastructure.profiling import (
    contact_drag_profiling_enabled,
    vertex_move_profiling_enabled,
    vertex_move_top_lines,
    write_profile_report,
)
from contour.graphics.geometry import (
    _snap_to_45,
)
from contour.graphics.tools import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode


class CycleActiveToolModeMixin:
    def _cycle_active_tool_mode(self) -> bool:
        if self._tool == EditorTool.ADD_POLYGON:
            if self._editor_scene.has_pending_polygon() or self._drag_kind == "rect_polygon":
                return False
            next_mode = (
                PolygonCreateMode.RECTANGLE
                if self._polygon_create_mode == PolygonCreateMode.POINTS
                else PolygonCreateMode.POINTS
            )
            self.set_polygon_create_mode(next_mode)
            return True
        if self._tool == EditorTool.BRUSH:
            order = [BrushMode.FREEFORM, BrushMode.ANGLED]
            index = order.index(self._brush_mode) if self._brush_mode in order else 0
            self.set_brush_mode(order[(index + 1) % len(order)])
            return True
        if self._tool == EditorTool.TRACE_PEN:
            order = [BrushMode.FREEFORM, BrushMode.ANGLED]
            index = order.index(self._trace_mode) if self._trace_mode in order else 0
            self.set_trace_mode(order[(index + 1) % len(order)])
            return True
        if self._tool == EditorTool.DELETE_VERTEX:
            next_mode = (
                DeleteVertexMode.AREA
                if self._delete_vertex_mode == DeleteVertexMode.SINGLE
                else DeleteVertexMode.SINGLE
            )
            self.set_delete_vertex_mode(next_mode)
            return True
        return False

    def _start_brush_drag(self, scene_pos: QPointF, *, erase: bool) -> None:
        self._drag_erases = bool(erase)
        self._drag_kind = "brush"
        self._drag_start_scene_pos = scene_pos
        self._editor_scene.start_pending_polygon(for_brush=True)
        self._editor_scene.set_pending_path_width(self._brush_thickness, cosmetic=False)
        self._append_brush_point(scene_pos)

    def _commit_brush_drag(self, release_pos: QPointF) -> None:
        if self._brush_mode == BrushMode.ANGLED and self._drag_start_scene_pos is not None:
            end_pos = release_pos
            if self._brush_pan_guard:
                # Panning changes the release coordinate, not the drawn endpoint.
                end_pos = self._last_pointer_scene_pos or self._drag_start_scene_pos
            end_point = _snap_to_45(self._drag_start_scene_pos, end_pos)
            brush_points = [
                (self._drag_start_scene_pos.x(), self._drag_start_scene_pos.y()),
                (end_point.x(), end_point.y()),
            ]
        else:
            if not self._brush_pan_guard:
                self._append_brush_point(release_pos)
            brush_points = self._editor_scene.pending_points_snapshot()
        self._brush_pan_guard = False
        self._editor_scene.add_brush_stroke(brush_points, self._brush_thickness, erase=self._drag_erases)

    def _append_trace_point(self, scene_pos: QPointF, *, erase: bool, snap: bool = False) -> None:
        if self._editor_scene.has_pending_polygon():
            if self._pending_polygon_erases is None:
                self._pending_polygon_erases = bool(erase)
            elif bool(erase) != self._pending_polygon_erases:
                self._finish_pending_trace()
                return
        else:
            self._pending_polygon_erases = bool(erase)
            self._editor_scene.start_pending_polygon(for_brush=True)
            self._editor_scene.set_pending_path_width(self._trace_width, cosmetic=False)

        target = scene_pos
        last_point = self._editor_scene.pending_last_point()
        if (snap or self._trace_mode == BrushMode.ANGLED) and last_point is not None:
            target = _snap_to_45(last_point, scene_pos)
        if last_point is not None and hypot(target.x() - last_point.x(), target.y() - last_point.y()) < 1.0:
            return
        self._editor_scene.append_pending_point(target)

    def _finish_pending_trace(self) -> None:
        points = self._editor_scene.pending_points_snapshot()
        self._editor_scene.add_trace_stroke(points, self._trace_width, erase=bool(self._pending_polygon_erases))
        self._pending_polygon_erases = None

    def _update_tool_cursors(self) -> None:
        self._editor_scene.set_brush_cursor(
            self._last_pointer_scene_pos,
            self._trace_width if self._tool == EditorTool.TRACE_PEN else self._brush_thickness,
            self._tool in (EditorTool.BRUSH, EditorTool.TRACE_PEN),
        )
        self._editor_scene.set_via_cursor(
            self._last_pointer_scene_pos,
            self._via_width,
            self._via_height,
            self._tool == EditorTool.ADD_VIA,
        )

    def _finish_pending_polygon(self) -> None:
        if self._pending_polygon_erases:
            self._editor_scene.subtract_pending_polygon()
        else:
            self._editor_scene.finish_pending_polygon()
        self._pending_polygon_erases = None
        self._emit_effective_polygon_create_mode_changed()

    def _ruler_target(self, start: QPointF, target: QPointF, modifiers: Qt.KeyboardModifier) -> QPointF:
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return _snap_to_45(start, target)
        return QPointF(target)

    def _format_ruler_measurement(self, start: QPointF, end: QPointF) -> str:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        distance = hypot(dx, dy)
        return f"L={distance:.1f}px, dX={dx:.1f}, dY={dy:.1f}"

    def _start_contact_drag_profile(self, polygon_id: int | None) -> None:
        if (
            not contact_drag_profiling_enabled()
            or not self._editor_scene.polygon_is_contact(polygon_id)
            or polygon_id is None
        ):
            return
        if self._contact_drag_profile is not None:
            self._finish_contact_drag_profile("superseded", commit_ms=0.0)
        contact_count = self._editor_scene.contact_count()
        self._contact_drag_profile = ContactDragProfile.begin(
            polygon_id=polygon_id,
            contact_count=contact_count,
        )
        write_profile_report(
            "[contour contact drag profiling] "
            f"started polygon_id={polygon_id} "
            f"contacts={contact_count}",
        )

    def _finish_contact_drag_profile(
        self,
        status: str,
        *,
        commit_ms: float,
    ) -> None:
        profile = self._contact_drag_profile
        if profile is None:
            return
        self._contact_drag_profile = None
        profile.finish(commit_ms=commit_ms)
        write_profile_report(profile.format_summary(status=status), profile.format_stats())

    def _emit_vertex_move_profile(
        self,
        timings_ms: dict[str, float],
        *,
        polygon_count: int,
        vertex_count: int,
        profiler: cProfile.Profile | None,
    ) -> None:
        if not vertex_move_profiling_enabled():
            return
        total_ms = timings_ms.get("total_wall", sum(timings_ms.values()))
        detail = " ".join(
            f"{name}={elapsed:.3f}ms" for name, elapsed in timings_ms.items() if name != "total_wall"
        )
        message = (
            f"[contour vertex profiling] total={total_ms:.3f}ms polygons={polygon_count} "
            f"vertices={vertex_count} {detail}"
        )
        reports = [message]
        if profiler is None:
            write_profile_report(*reports)
            return
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        top_lines = vertex_move_top_lines()
        stats.print_stats(top_lines)
        report = stream.getvalue()
        reports.extend((f"[contour vertex profiling stats] top={top_lines}", report))
        write_profile_report(*reports)

    def _cancel_move_vertex_tool_profile_finish(self) -> None:
        self._move_vertex_tool_profile_generation += 1
        self._move_vertex_tool_profile_paint_started_at = None

    def _schedule_move_vertex_tool_profile_finish(self) -> None:
        self._move_vertex_tool_profile_generation += 1
        generation = self._move_vertex_tool_profile_generation
        self._move_vertex_tool_profile_paint_started_at = perf_counter()
        QTimer.singleShot(
            0,
            lambda: self._finish_move_vertex_tool_profile_after_paint(generation),
        )
