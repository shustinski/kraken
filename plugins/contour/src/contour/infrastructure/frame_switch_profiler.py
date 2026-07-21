"""End-to-end profiling for Contour image/frame switches until the UI is interactive."""

from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from .profiling import (
    frame_switch_idle_polls,
    frame_switch_profiling_enabled,
    frame_switch_top_lines,
    try_disable_profiler,
    try_enable_profiler,
)

MAX_IDLE_POLLS = frame_switch_idle_polls()


@dataclass
class FrameSwitchProfile:
    image_path: str
    started_at: float
    generation: int
    profiler: cProfile.Profile = field(default_factory=cProfile.Profile)
    worker_profilers: list[tuple[str, cProfile.Profile]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    pending_since: dict[str, float] = field(default_factory=dict)
    poll_count: int = 0
    profiling_active: bool = False
    main_stats_skipped: bool = False

    @classmethod
    def begin(cls, image_path: str, *, generation: int) -> FrameSwitchProfile:
        return cls(
            image_path=str(Path(image_path)),
            started_at=perf_counter(),
            generation=int(generation),
        )

    def enable_main_profiler(self) -> bool:
        if self.profiling_active or self.main_stats_skipped:
            return self.profiling_active
        if try_enable_profiler(self.profiler):
            self.profiling_active = True
            return True
        self.main_stats_skipped = True
        return False

    def disable_main_profiler(self) -> cProfile.Profile:
        if self.profiling_active:
            try_disable_profiler(self.profiler)
            self.profiling_active = False
        return self.profiler

    def merge_timings(self, timings_ms: dict[str, float]) -> None:
        for key, value in timings_ms.items():
            if key == "total_wall":
                continue
            self.timings_ms[key] = float(value)

    def note_timing(self, name: str, elapsed_ms: float) -> None:
        self.timings_ms[name] = float(elapsed_ms)

    def mark_pending(self, name: str) -> None:
        self.pending_since[name] = perf_counter()

    def complete_pending(self, name: str, *, suffix: str = "") -> None:
        started = self.pending_since.pop(name, None)
        if started is None:
            return
        label = f"{name}{suffix}" if suffix else name
        self.timings_ms[label] = (perf_counter() - started) * 1000.0

    def attach_worker_profile(self, label: str, profiler: cProfile.Profile) -> None:
        self.worker_profilers.append((label, profiler))

    def total_wall_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000.0

    def format_summary(
        self,
        *,
        polygon_count: int,
        cif_path: str | None,
        vectors_only: bool,
        failed: bool,
        interactive: bool,
    ) -> str:
        mode = "vector" if vectors_only else "image+vector"
        status = "failed" if failed else ("interactive" if interactive else "partial")
        detail = " ".join(
            f"{name}={elapsed:.3f}ms"
            for name, elapsed in sorted(self.timings_ms.items(), key=lambda item: item[0])
        )
        skipped = " main_cprofile_skipped=yes" if self.main_stats_skipped else ""
        pending = f" pending={','.join(sorted(self.pending_since))}" if self.pending_since else ""
        return (
            f"[contour frame switch profiling] mode={mode} status={status} "
            f"total={self.total_wall_ms():.3f}ms polygons={polygon_count} "
            f"image={Path(self.image_path).name} "
            f"cif={Path(cif_path).name if cif_path else '<none>'} polls={self.poll_count}{skipped}{pending} {detail}"
        )

    def format_stats(self, profiler: cProfile.Profile, *, title: str) -> str:
        top_lines = frame_switch_top_lines()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats(top_lines)
        return f"[contour frame switch profiling stats] {title} top={top_lines}\n{stream.getvalue()}"


def profile_callable(label: str, profile: FrameSwitchProfile | None, fn, /):
    """Profile background work; uses cProfile only when the main-thread slot is free."""
    if profile is None:
        return fn()
    if profile.profiling_active:
        started = perf_counter()
        try:
            return fn()
        finally:
            profile.note_timing(f"worker_{label}_wall", (perf_counter() - started) * 1000.0)
    worker = cProfile.Profile()
    try:
        if try_enable_profiler(worker):
            try:
                return fn()
            finally:
                try_disable_profiler(worker)
        started = perf_counter()
        try:
            return fn()
        finally:
            profile.note_timing(f"worker_{label}_wall", (perf_counter() - started) * 1000.0)
    finally:
        if worker.getstats():
            profile.attach_worker_profile(label, worker)
