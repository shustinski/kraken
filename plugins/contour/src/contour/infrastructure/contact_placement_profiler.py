"""End-to-end profiling for a manually placed contact."""

from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import dataclass, field
from time import perf_counter

from .profiling import (
    contact_copy_top_lines,
    contact_deletion_top_lines,
    contact_multi_selection_top_lines,
    contact_paste_top_lines,
    contact_placement_top_lines,
    contact_redo_top_lines,
    contact_undo_top_lines,
    try_disable_profiler,
    try_enable_profiler,
)


@dataclass
class ContactPlacementProfile:
    """Collect UI- and worker-thread stats for one contact placement."""

    started_at: float = field(default_factory=perf_counter)
    main_profiler: cProfile.Profile = field(default_factory=cProfile.Profile)
    worker_profilers: list[cProfile.Profile] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    main_active: bool = False
    main_stats_skipped: bool = False
    preview_request_id: int | None = None
    waiting_for_preview: bool = False
    action: str = "placement"
    contact_count: int = 0

    @classmethod
    def begin(cls, *, action: str = "placement") -> ContactPlacementProfile:
        session = cls(action=str(action))
        if try_enable_profiler(session.main_profiler):
            session.main_active = True
        else:
            session.main_stats_skipped = True
        return session

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000.0

    def note(self, name: str) -> None:
        self.timings_ms[name] = self.elapsed_ms()

    def attach_worker(self, profiler: cProfile.Profile, wall_ms: float) -> None:
        if profiler.getstats():
            self.worker_profilers.append(profiler)
        self.timings_ms["preview_worker_wall"] = float(wall_ms)

    def stop(self) -> None:
        if self.main_active:
            try_disable_profiler(self.main_profiler)
            self.main_active = False

    def format_summary(self, *, status: str) -> str:
        phases = " ".join(
            f"{name}={value:.3f}ms"
            for name, value in sorted(self.timings_ms.items())
        )
        skipped = " main_cprofile_skipped=yes" if self.main_stats_skipped else ""
        count = f" contacts={self.contact_count}" if self.contact_count > 0 else ""
        return (
            f"[contour contact {self.action.replace('_', ' ')} profiling] status={status} "
            f"total={self.elapsed_ms():.3f}ms{count}{skipped}"
            f"{(' ' + phases) if phases else ''}"
        )

    def format_stats(self) -> str:
        if self.action == "deletion":
            top_lines = contact_deletion_top_lines()
        elif self.action == "copy":
            top_lines = contact_copy_top_lines()
        elif self.action == "paste":
            top_lines = contact_paste_top_lines()
        elif self.action == "multi_selection":
            top_lines = contact_multi_selection_top_lines()
        elif self.action == "undo":
            top_lines = contact_undo_top_lines()
        elif self.action == "redo":
            top_lines = contact_redo_top_lines()
        else:
            top_lines = contact_placement_top_lines()
        prefix = f"[contour contact {self.action.replace('_', ' ')} profiling stats]"
        profiles = [
            profile
            for profile in (self.main_profiler, *self.worker_profilers)
            if profile.getstats()
        ]
        if not profiles:
            return (
                f"{prefix} no cProfile data collected"
            )
        stream = io.StringIO()
        stats = pstats.Stats(profiles[0], stream=stream)
        for profile in profiles[1:]:
            stats.add(profile)
        stats.sort_stats("cumtime").print_stats(top_lines)
        return (
            f"{prefix} sort=cumulative top={top_lines}\n{stream.getvalue()}"
        )
