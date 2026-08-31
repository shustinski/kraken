"""End-to-end profiling for one editor polygon mutation."""

from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import dataclass, field
from time import perf_counter

from .profiling import (
    polygon_change_top_lines,
    try_disable_profiler,
    try_enable_profiler,
)


@dataclass
class PolygonChangeProfile:
    operation: str
    polygon_count_before: int
    started_at: float = field(default_factory=perf_counter)
    timings_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    profiler: cProfile.Profile = field(default_factory=cProfile.Profile)
    profiler_active: bool = False
    main_stats_skipped: bool = False

    @classmethod
    def begin(cls, *, operation: str, polygon_count_before: int) -> PolygonChangeProfile:
        session = cls(
            operation=str(operation),
            polygon_count_before=max(0, int(polygon_count_before)),
        )
        if try_enable_profiler(session.profiler):
            session.profiler_active = True
        else:
            session.main_stats_skipped = True
        return session

    def elapsed_ms(self) -> float:
        return max(0.0, (perf_counter() - self.started_at) * 1000.0)

    def note_phase(self, name: str, started_at: float) -> None:
        elapsed_ms = max(0.0, (perf_counter() - float(started_at)) * 1000.0)
        self.timings_ms[str(name)] = float(self.timings_ms.get(str(name), 0.0)) + elapsed_ms

    def set_metadata(self, **values: object) -> None:
        self.metadata.update(values)

    def finish(self) -> None:
        self.timings_ms.setdefault("total_wall", self.elapsed_ms())
        if self.profiler_active:
            try_disable_profiler(self.profiler)
            self.profiler_active = False

    def format_summary(self, *, status: str) -> str:
        phases = " ".join(
            f"{name}={value:.3f}ms"
            for name, value in sorted(self.timings_ms.items())
            if name != "total_wall"
        )
        metadata = " ".join(
            f"{name}={value}"
            for name, value in sorted(self.metadata.items())
        )
        skipped = " main_cprofile_skipped=yes" if self.main_stats_skipped else ""
        total_ms = float(self.timings_ms.get("total_wall", self.elapsed_ms()))
        return (
            f"[contour polygon change profiling] status={status} "
            f"operation={self.operation} total={total_ms:.3f}ms "
            f"polygons_before={self.polygon_count_before}{skipped}"
            f"{(' ' + metadata) if metadata else ''}"
            f"{(' ' + phases) if phases else ''}"
        )

    def format_stats(self) -> str:
        prefix = "[contour polygon change profiling stats]"
        if not self.profiler.getstats():
            return f"{prefix} no cProfile data collected"
        top_lines = polygon_change_top_lines()
        stream = io.StringIO()
        pstats.Stats(self.profiler, stream=stream).sort_stats("cumtime").print_stats(top_lines)
        return f"{prefix} sort=cumulative top={top_lines}\n{stream.getvalue()}"
