"""editor_view: commit delete vertices in."""
from __future__ import annotations
import cProfile
import io
import pstats
from time import perf_counter
from PyQt6.QtCore import (
    QPointF,
    QRectF,
)
from ...infrastructure.profiling import (
    delete_area_profiling_enabled,
    delete_area_top_lines,
    try_disable_profiler,
    try_enable_profiler,
    write_profile_report,
)


class CommitDeleteVerticesInMixin:
    def _finish_move_vertex_tool_profile_after_paint(self, generation: int) -> None:
        if generation != self._move_vertex_tool_profile_generation:
            return
        profile = self._move_vertex_tool_profile
        if profile is None:
            return
        paint_started_at = self._move_vertex_tool_profile_paint_started_at
        if paint_started_at is not None:
            profile.note_timing(
                "paint",
                (perf_counter() - paint_started_at) * 1000.0,
            )
        self._finish_move_vertex_tool_profile("displayed")

    def _finish_move_vertex_tool_profile(self, status: str) -> None:
        profile = self._move_vertex_tool_profile
        if profile is None:
            return
        self._move_vertex_tool_profile = None
        self._move_vertex_tool_profile_paint_started_at = None
        profile.note_timing("total_wall", profile.total_wall_ms())
        profile.finish()
        write_profile_report(profile.format_summary(status=status), profile.format_stats())

    def _commit_delete_vertices_in_area(self, rect: QRectF) -> None:
        profiling_enabled = delete_area_profiling_enabled()
        profile_timings: dict[str, float] = {}
        profile_total_start = perf_counter()
        profiler = cProfile.Profile() if profiling_enabled else None
        profiler_enabled = False
        if profiler is not None:
            profiler_enabled = try_enable_profiler(profiler)
        try:
            polygon_hits = vertex_hits = vertex_total = 0
            if profiling_enabled:
                phase_start = perf_counter()
                polygon_hits, vertex_hits, vertex_total = self._delete_area_profile_counts(rect)
                profile_timings["count"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
            deleted = self._editor_scene.delete_vertices_in_rect(rect)
            if profiling_enabled:
                profile_timings["delete"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
            self._editor_scene.clear_preview_rect()
            if profiling_enabled:
                profile_timings["clear_preview"] = (perf_counter() - phase_start) * 1000.0
                profile_timings["total_wall"] = (perf_counter() - profile_total_start) * 1000.0
                self._emit_delete_area_profile(
                    profile_timings,
                    polygon_hits=polygon_hits,
                    vertex_hits=vertex_hits,
                    vertex_total=vertex_total,
                    deleted=deleted,
                    profiler=profiler if profiler_enabled else None,
                )
        finally:
            if profiler_enabled and profiler is not None:
                try_disable_profiler(profiler)

    def _delete_area_profile_counts(self, rect: QRectF) -> tuple[int, int, int]:
        normalized = rect.normalized()
        polygon_hits = 0
        vertex_hits = 0
        vertex_total = 0
        for polygon in self._editor_scene.get_polygons():
            vertex_total += len(polygon.points)
            matched = sum(
                1
                for x_coord, y_coord in polygon.points
                if normalized.contains(QPointF(x_coord, y_coord))
            )
            if matched:
                polygon_hits += 1
                vertex_hits += matched
        return polygon_hits, vertex_hits, vertex_total

    def _emit_delete_area_profile(
        self,
        timings_ms: dict[str, float],
        *,
        polygon_hits: int,
        vertex_hits: int,
        vertex_total: int,
        deleted: int,
        profiler: cProfile.Profile | None,
    ) -> None:
        if not delete_area_profiling_enabled():
            return
        total_ms = timings_ms.get("total_wall", sum(timings_ms.values()))
        detail = " ".join(
            f"{name}={elapsed:.3f}ms" for name, elapsed in timings_ms.items() if name != "total_wall"
        )
        message = (
            f"[contour delete-area profiling] total={total_ms:.3f}ms "
            f"polygons_hit={polygon_hits} vertices_hit={vertex_hits} "
            f"vertices_total={vertex_total} deleted={deleted} {detail}"
        )
        reports = [message]
        if profiler is None:
            write_profile_report(*reports)
            return
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        top_lines = delete_area_top_lines()
        stats.print_stats(top_lines)
        report = stream.getvalue()
        reports.extend((f"[contour delete-area profiling stats] top={top_lines}", report))
        write_profile_report(*reports)
