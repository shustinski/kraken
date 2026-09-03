"""Profile the standalone Contour launch until the first Qt event-loop turn."""

from __future__ import annotations

import cProfile
import io
import pstats
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, TypeVar

from .profiling import (
    startup_profiling_enabled,
    startup_top_lines,
    try_disable_profiler,
    try_enable_profiler,
    write_profile_report,
)

_ResultT = TypeVar("_ResultT")


@dataclass
class StartupProfile:
    started_at: float
    profiler: cProfile.Profile = field(default_factory=cProfile.Profile)
    timings_ms: dict[str, float] = field(default_factory=dict)
    profiling_active: bool = False
    stats_skipped: bool = False
    finished: bool = False

    @classmethod
    def begin(cls) -> StartupProfile | None:
        if not startup_profiling_enabled():
            return None
        profile = cls(started_at=perf_counter())
        if try_enable_profiler(profile.profiler):
            profile.profiling_active = True
        else:
            profile.stats_skipped = True
        return profile

    def measure(self, name: str, operation: Callable[[], _ResultT], /) -> _ResultT:
        started_at = perf_counter()
        try:
            return operation()
        finally:
            self.timings_ms[name] = (perf_counter() - started_at) * 1000.0

    def mark_interval(self, name: str, started_at: float) -> None:
        self.timings_ms[name] = (perf_counter() - started_at) * 1000.0

    def finish(self, *, status: Literal["interactive", "failed", "stopped"]) -> None:
        if self.finished:
            return
        self.finished = True
        if self.profiling_active:
            try_disable_profiler(self.profiler)
            self.profiling_active = False

        total_ms = (perf_counter() - self.started_at) * 1000.0
        details = " ".join(f"{name}={elapsed:.3f}ms" for name, elapsed in self.timings_ms.items())
        skipped = " cprofile_skipped=yes" if self.stats_skipped else ""
        summary = f"[contour startup profiling] status={status} total={total_ms:.3f}ms{skipped} {details}".rstrip()

        reports = [summary]
        if self.profiler.getstats():
            top_lines = startup_top_lines()
            stream = io.StringIO()
            pstats.Stats(self.profiler, stream=stream).sort_stats("cumtime").print_stats(top_lines)
            reports.append(f"[contour startup profiling stats] top={top_lines}\n{stream.getvalue()}")
        write_profile_report(*reports)
