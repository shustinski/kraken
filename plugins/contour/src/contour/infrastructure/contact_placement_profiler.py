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
    contact_drag_top_lines,
    contact_multi_selection_top_lines,
    contact_paste_top_lines,
    contact_placement_top_lines,
    contact_redo_top_lines,
    contact_undo_top_lines,
    image_recognition_top_lines,
    move_vertex_tool_top_lines,
    scene_zoom_top_lines,
    try_disable_profiler,
    try_enable_profiler,
)


@dataclass
class SceneZoomProfile:
    initial_zoom: float
    target_zoom: float
    started_at: float = field(default_factory=perf_counter)
    profiler: cProfile.Profile = field(default_factory=cProfile.Profile)
    profiler_active: bool = False
    main_stats_skipped: bool = False
    frame_durations_ms: list[float] = field(default_factory=list)
    frame_intervals_ms: list[float] = field(default_factory=list)
    last_frame_at: float | None = None

    @classmethod
    def begin(
        cls,
        *,
        initial_zoom: float,
        target_zoom: float,
    ) -> SceneZoomProfile:
        session = cls(
            initial_zoom=max(0.0, float(initial_zoom)),
            target_zoom=max(0.0, float(target_zoom)),
        )
        if try_enable_profiler(session.profiler):
            session.profiler_active = True
        else:
            session.main_stats_skipped = True
        return session

    def update_target(self, target_zoom: float) -> None:
        self.target_zoom = max(0.0, float(target_zoom))

    def record_frame(self, frame_started_at: float) -> None:
        completed_at = perf_counter()
        self.frame_durations_ms.append(
            max(0.0, (completed_at - frame_started_at) * 1000.0)
        )
        if self.last_frame_at is not None:
            self.frame_intervals_ms.append(
                max(0.0, (completed_at - self.last_frame_at) * 1000.0)
            )
        self.last_frame_at = completed_at

    def finish(self) -> None:
        if self.profiler_active:
            try_disable_profiler(self.profiler)
            self.profiler_active = False

    def elapsed_ms(self) -> float:
        return max(0.0, (perf_counter() - self.started_at) * 1000.0)

    def _fps(self) -> float:
        if self.frame_intervals_ms:
            mean_interval = sum(self.frame_intervals_ms) / len(
                self.frame_intervals_ms
            )
            return 1000.0 / mean_interval if mean_interval > 0.0 else 0.0
        elapsed = self.elapsed_ms()
        return (
            len(self.frame_durations_ms) * 1000.0 / elapsed
            if elapsed > 0.0
            else 0.0
        )

    def format_summary(self, *, status: str, final_zoom: float) -> str:
        durations = sorted(self.frame_durations_ms)
        average_ms = sum(durations) / len(durations) if durations else 0.0
        p95_index = max(0, min(len(durations) - 1, round(len(durations) * 0.95) - 1))
        p95_ms = durations[p95_index] if durations else 0.0
        maximum_ms = durations[-1] if durations else 0.0
        skipped = " main_cprofile_skipped=yes" if self.main_stats_skipped else ""
        return (
            f"[contour scene zoom profiling] status={status} "
            f"total={self.elapsed_ms():.3f}ms "
            f"zoom={self.initial_zoom:.4f}->{max(0.0, float(final_zoom)):.4f} "
            f"target={self.target_zoom:.4f} frames={len(durations)} "
            f"fps={self._fps():.2f} frame_avg={average_ms:.3f}ms "
            f"frame_p95={p95_ms:.3f}ms frame_max={maximum_ms:.3f}ms{skipped}"
        )

    def format_stats(self) -> str:
        prefix = "[contour scene zoom profiling stats]"
        if not self.profiler.getstats():
            return f"{prefix} no cProfile data collected"
        top_lines = scene_zoom_top_lines()
        stream = io.StringIO()
        pstats.Stats(self.profiler, stream=stream).sort_stats(
            "cumtime"
        ).print_stats(top_lines)
        return f"{prefix} sort=cumulative top={top_lines}\n{stream.getvalue()}"


@dataclass
class MoveVertexToolActivationProfile:
    polygon_count: int
    vertex_count: int
    started_at: float = field(default_factory=perf_counter)
    profiler: cProfile.Profile = field(default_factory=cProfile.Profile)
    profiler_active: bool = False
    main_stats_skipped: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def begin(
        cls,
        *,
        polygon_count: int,
        vertex_count: int,
    ) -> MoveVertexToolActivationProfile:
        session = cls(
            polygon_count=max(0, int(polygon_count)),
            vertex_count=max(0, int(vertex_count)),
        )
        if try_enable_profiler(session.profiler):
            session.profiler_active = True
        else:
            session.main_stats_skipped = True
        return session

    def note_timing(self, name: str, elapsed_ms: float) -> None:
        self.timings_ms[name] = max(0.0, float(elapsed_ms))

    def finish(self) -> None:
        if self.profiler_active:
            try_disable_profiler(self.profiler)
            self.profiler_active = False

    def total_wall_ms(self) -> float:
        return max(0.0, (perf_counter() - self.started_at) * 1000.0)

    def format_summary(self, *, status: str) -> str:
        total_ms = self.timings_ms.get("total_wall", self.total_wall_ms())
        detail = " ".join(
            f"{name}={elapsed:.3f}ms"
            for name, elapsed in self.timings_ms.items()
            if name != "total_wall"
        )
        skipped = " main_cprofile_skipped=yes" if self.main_stats_skipped else ""
        return (
            f"[contour move-vertex-tool profiling] status={status} "
            f"total={total_ms:.3f}ms polygons={self.polygon_count} "
            f"vertices={self.vertex_count} {detail}{skipped}"
        )

    def format_stats(self) -> str:
        prefix = "[contour move-vertex-tool profiling stats]"
        if not self.profiler.getstats():
            return f"{prefix} no cProfile data collected"
        top_lines = move_vertex_tool_top_lines()
        stream = io.StringIO()
        pstats.Stats(self.profiler, stream=stream).sort_stats(
            "cumtime"
        ).print_stats(top_lines)
        return f"{prefix} sort=cumulative top={top_lines}\n{stream.getvalue()}"


@dataclass
class ContactDragProfile:
    polygon_id: int
    contact_count: int
    started_at: float = field(default_factory=perf_counter)
    profiler: cProfile.Profile = field(default_factory=cProfile.Profile)
    profiler_active: bool = False
    main_stats_skipped: bool = False
    frame_durations_ms: list[float] = field(default_factory=list)
    frame_intervals_ms: list[float] = field(default_factory=list)
    last_frame_at: float | None = None
    commit_ms: float = 0.0

    @classmethod
    def begin(
        cls,
        *,
        polygon_id: int,
        contact_count: int,
    ) -> ContactDragProfile:
        session = cls(
            polygon_id=int(polygon_id),
            contact_count=max(0, int(contact_count)),
        )
        if try_enable_profiler(session.profiler):
            session.profiler_active = True
        else:
            session.main_stats_skipped = True
        return session

    def record_frame(self, event_started_at: float) -> None:
        completed_at = perf_counter()
        self.frame_durations_ms.append(
            max(0.0, (completed_at - event_started_at) * 1000.0)
        )
        if self.last_frame_at is not None:
            self.frame_intervals_ms.append(
                max(0.0, (completed_at - self.last_frame_at) * 1000.0)
            )
        self.last_frame_at = completed_at

    def finish(self, *, commit_ms: float) -> None:
        self.commit_ms = max(0.0, float(commit_ms))
        if self.profiler_active:
            try_disable_profiler(self.profiler)
            self.profiler_active = False

    def elapsed_ms(self) -> float:
        return max(0.0, (perf_counter() - self.started_at) * 1000.0)

    def _fps(self) -> float:
        if self.frame_intervals_ms:
            mean_interval = sum(self.frame_intervals_ms) / len(
                self.frame_intervals_ms
            )
            return 1000.0 / mean_interval if mean_interval > 0.0 else 0.0
        elapsed = self.elapsed_ms()
        return (
            len(self.frame_durations_ms) * 1000.0 / elapsed
            if elapsed > 0.0
            else 0.0
        )

    def format_summary(self, *, status: str) -> str:
        durations = sorted(self.frame_durations_ms)
        average_ms = sum(durations) / len(durations) if durations else 0.0
        p95_index = max(0, min(len(durations) - 1, round(len(durations) * 0.95) - 1))
        p95_ms = durations[p95_index] if durations else 0.0
        maximum_ms = durations[-1] if durations else 0.0
        skipped = " main_cprofile_skipped=yes" if self.main_stats_skipped else ""
        return (
            f"[contour contact drag profiling] status={status} "
            f"total={self.elapsed_ms():.3f}ms polygon_id={self.polygon_id} "
            f"contacts={self.contact_count} frames={len(durations)} "
            f"fps={self._fps():.2f} frame_avg={average_ms:.3f}ms "
            f"frame_p95={p95_ms:.3f}ms frame_max={maximum_ms:.3f}ms "
            f"commit={self.commit_ms:.3f}ms{skipped}"
        )

    def format_stats(self) -> str:
        prefix = "[contour contact drag profiling stats]"
        if not self.profiler.getstats():
            return f"{prefix} no cProfile data collected"
        top_lines = contact_drag_top_lines()
        stream = io.StringIO()
        pstats.Stats(self.profiler, stream=stream).sort_stats(
            "cumtime"
        ).print_stats(top_lines)
        return (
            f"{prefix} sort=cumulative top={top_lines}\n"
            f"{stream.getvalue()}"
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

    def restart_main_profiler(self) -> None:
        self.stop()
        self.main_profiler.clear()
        if try_enable_profiler(self.main_profiler):
            self.main_active = True
            self.main_stats_skipped = False
        else:
            self.main_stats_skipped = True

    def resume_main_profiler(self) -> None:
        if self.main_active:
            return
        if try_enable_profiler(self.main_profiler):
            self.main_active = True
        else:
            self.main_stats_skipped = True

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


@dataclass
class ImageRecognitionProfile(ContactPlacementProfile):
    image_path: str = ""
    recognition_mode: str = ""
    polygon_count: int = 0

    def attach_worker(self, profiler: cProfile.Profile, wall_ms: float) -> None:
        if profiler.getstats():
            self.worker_profilers.append(profiler)
        self.timings_ms["worker_wall"] = float(wall_ms)

    @classmethod
    def begin(
        cls,
        *,
        image_path: str,
        recognition_mode: str,
    ) -> ImageRecognitionProfile:
        session = cls(
            action="recognition",
            image_path=str(image_path),
            recognition_mode=str(recognition_mode),
        )
        if try_enable_profiler(session.main_profiler):
            session.main_active = True
        else:
            session.main_stats_skipped = True
        return session

    def format_summary(self, *, status: str) -> str:
        phases = " ".join(
            f"{name}={value:.3f}ms"
            for name, value in sorted(self.timings_ms.items())
        )
        skipped = " main_cprofile_skipped=yes" if self.main_stats_skipped else ""
        return (
            f"[contour image recognition profiling] status={status} "
            f"total={self.elapsed_ms():.3f}ms mode={self.recognition_mode} "
            f"polygons={self.polygon_count} image={self.image_path!r}{skipped}"
            f"{(' ' + phases) if phases else ''}"
        )

    def format_stats(self) -> str:
        prefix = "[contour image recognition profiling stats]"
        profiles = [
            profile
            for profile in (self.main_profiler, *self.worker_profilers)
            if profile.getstats()
        ]
        if not profiles:
            return f"{prefix} no cProfile data collected"
        stream = io.StringIO()
        stats = pstats.Stats(profiles[0], stream=stream)
        for profile in profiles[1:]:
            stats.add(profile)
        top_lines = image_recognition_top_lines()
        stats.sort_stats("cumtime").print_stats(top_lines)
        return f"{prefix} sort=cumulative top={top_lines}\n{stream.getvalue()}"
