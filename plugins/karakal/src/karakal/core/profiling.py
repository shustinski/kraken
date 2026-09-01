"""Bounded, dependency-free hierarchical profiling for normal Karakal runs."""

from __future__ import annotations

import csv
import heapq
import json
import os
import platform
import sys
import threading
import uuid
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import Iterator

import numpy as np

from .performance import PerformanceConfig, ProfilingMode


_RESERVOIR_LIMIT = 512
_SLOW_FRAME_LIMIT = 100
_TRACE_EVENT_LIMIT = 20_000


@dataclass(slots=True)
class StageAggregate:
    name: str
    calls: int = 0
    frame_count: int = 0
    batch_count: int = 0
    inclusive_ns: int = 0
    self_ns: int = 0
    cpu_ns: int = 0
    minimum_ns: int = 0
    maximum_ns: int = 0
    mean_ns: float = 0.0
    m2_ns: float = 0.0
    errors: int = 0
    cancellations: int = 0
    samples_ns: list[int] = field(default_factory=list)

    def add(
        self,
        inclusive_ns: int,
        self_ns: int,
        cpu_ns: int,
        *,
        frame_count: int,
        batch_count: int,
        error: bool,
        cancelled: bool,
    ) -> None:
        duration = max(0, int(inclusive_ns))
        self.calls += 1
        self.frame_count += max(0, int(frame_count))
        self.batch_count += max(0, int(batch_count))
        self.inclusive_ns += duration
        self.self_ns += max(0, int(self_ns))
        self.cpu_ns += max(0, int(cpu_ns))
        self.minimum_ns = duration if self.calls == 1 else min(self.minimum_ns, duration)
        self.maximum_ns = max(self.maximum_ns, duration)
        delta = duration - self.mean_ns
        self.mean_ns += delta / self.calls
        self.m2_ns += delta * (duration - self.mean_ns)
        self.errors += int(error)
        self.cancellations += int(cancelled)
        if len(self.samples_ns) < _RESERVOIR_LIMIT:
            self.samples_ns.append(duration)
        else:
            # Deterministic bounded sampling must not perturb the application's RNG.
            self.samples_ns[(self.calls - 1) % _RESERVOIR_LIMIT] = duration

    def merge(self, other: "StageAggregate") -> None:
        if other.calls <= 0:
            return
        previous_calls = self.calls
        combined_calls = previous_calls + other.calls
        if previous_calls == 0:
            self.mean_ns = other.mean_ns
            self.m2_ns = other.m2_ns
            self.minimum_ns = other.minimum_ns
            self.maximum_ns = other.maximum_ns
        else:
            delta = other.mean_ns - self.mean_ns
            self.mean_ns += delta * other.calls / combined_calls
            self.m2_ns += other.m2_ns + delta * delta * previous_calls * other.calls / combined_calls
            self.minimum_ns = min(self.minimum_ns, other.minimum_ns)
            self.maximum_ns = max(self.maximum_ns, other.maximum_ns)
        self.calls = combined_calls
        self.inclusive_ns += other.inclusive_ns
        self.self_ns += other.self_ns
        self.cpu_ns += other.cpu_ns
        self.frame_count += other.frame_count
        self.batch_count += other.batch_count
        self.errors += other.errors
        self.cancellations += other.cancellations
        combined_samples = self.samples_ns + other.samples_ns
        if len(combined_samples) <= _RESERVOIR_LIMIT:
            self.samples_ns = combined_samples
        else:
            step = len(combined_samples) / _RESERVOIR_LIMIT
            self.samples_ns = [combined_samples[int(index * step)] for index in range(_RESERVOIR_LIMIT)]

    def percentile_ms(self, percentile: float) -> float | None:
        if not self.samples_ns:
            return None
        return float(np.percentile(np.asarray(self.samples_ns, dtype=np.float64), percentile) / 1_000_000.0)

    def to_payload(self, total_ns: int) -> dict[str, object]:
        return {
            "name": self.name,
            "calls": self.calls,
            "frame_count": self.frame_count,
            "batch_count": self.batch_count,
            "total_ms": self.inclusive_ns / 1_000_000.0,
            "self_ms": self.self_ns / 1_000_000.0,
            "cpu_ms": self.cpu_ns / 1_000_000.0,
            "average_ms": self.mean_ns / 1_000_000.0 if self.calls else None,
            "minimum_ms": self.minimum_ns / 1_000_000.0 if self.calls else None,
            "maximum_ms": self.maximum_ns / 1_000_000.0 if self.calls else None,
            "median_ms": self.percentile_ms(50.0),
            "p90_ms": self.percentile_ms(90.0),
            "p95_ms": self.percentile_ms(95.0),
            "p99_ms": self.percentile_ms(99.0),
            "share": self.inclusive_ns / max(1, total_ns),
            "errors": self.errors,
            "cancellations": self.cancellations,
        }


@dataclass(frozen=True, slots=True)
class WorkerProfilePacket:
    worker_id: str
    pid: int
    stages: tuple[StageAggregate, ...]
    counters: tuple[tuple[str, int], ...]
    processed_frames: int
    processed_batches: int


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    run_id: str
    analysis_type: str
    mode: ProfilingMode
    started_at: str
    elapsed_ms: float
    current_stage: str | None
    processed_frames: int
    stages: tuple[dict[str, object], ...]
    counters: dict[str, int]
    slow_frames: tuple[dict[str, object], ...]
    workers: tuple[dict[str, object], ...]
    environment: dict[str, object]
    trace_events: tuple[dict[str, object], ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "analysis_type": self.analysis_type,
            "mode": self.mode.value,
            "started_at": self.started_at,
            "elapsed_ms": self.elapsed_ms,
            "current_stage": self.current_stage,
            "processed_frames": self.processed_frames,
            "stages": list(self.stages),
            "counters": dict(self.counters),
            "slow_frames": list(self.slow_frames),
            "workers": list(self.workers),
            "environment": dict(self.environment),
        }


@dataclass(slots=True)
class _ActiveStage:
    name: str
    started_ns: int
    cpu_started_ns: int
    child_ns: int = 0


class _StageContext(AbstractContextManager[None]):
    def __init__(
        self,
        profiler: "ProfilerRun",
        name: str,
        *,
        task_id: str | None,
        frame_id: str | None,
        frame_count: int,
        batch_count: int,
    ) -> None:
        self.profiler = profiler
        self.name = str(name)
        self.task_id = task_id
        self.frame_id = frame_id
        self.frame_count = int(frame_count)
        self.batch_count = int(batch_count)
        self.active: _ActiveStage | None = None

    def __enter__(self) -> None:
        if not self.profiler.enabled:
            return None
        self.active = self.profiler._enter_stage(self.name, self.task_id, self.frame_id)
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.active is not None:
            self.profiler._exit_stage(
                self.active,
                frame_id=self.frame_id,
                frame_count=self.frame_count,
                batch_count=self.batch_count,
                error=exc_value is not None,
                cancelled=exc_type is GeneratorExit,
            )
        return False


class ProfilerRun:
    """Thread-safe profiler whose disabled path only performs a boolean check."""

    def __init__(self, analysis_type: str, config: PerformanceConfig, *, run_id: str | None = None) -> None:
        self.analysis_type = str(analysis_type or "validation")
        self.config = config
        self.run_id = str(run_id or uuid.uuid4().hex)
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self._started_ns = perf_counter_ns()
        self._lock = threading.RLock()
        self._local = threading.local()
        self._stages: dict[str, StageAggregate] = {}
        self._counters: dict[str, int] = {}
        self._slow_frames: list[tuple[int, str, str]] = []
        self._worker_packets: dict[str, WorkerProfilePacket] = {}
        self._trace_events: list[dict[str, object]] = []
        self._trace_frame_ids: set[str] = set()
        self._current_stage: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.profiling_mode is not ProfilingMode.OFF

    def stage(
        self,
        name: str,
        *,
        task_id: str | None = None,
        frame_id: str | None = None,
        frame_count: int = 0,
        batch_count: int = 0,
    ) -> AbstractContextManager[None]:
        if not self.enabled:
            return _NULL_STAGE
        return _StageContext(
            self,
            name,
            task_id=task_id,
            frame_id=frame_id,
            frame_count=frame_count,
            batch_count=batch_count,
        )

    def _stack(self) -> list[_ActiveStage]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _enter_stage(self, name: str, task_id: str | None, frame_id: str | None) -> _ActiveStage:
        active = _ActiveStage(name, perf_counter_ns(), process_time_ns())
        self._stack().append(active)
        with self._lock:
            self._current_stage = name
            if self.config.profiling_mode is ProfilingMode.TRACE and len(self._trace_events) < _TRACE_EVENT_LIMIT:
                if (
                    frame_id is None
                    or frame_id in self._trace_frame_ids
                    or len(self._trace_frame_ids) < self.config.profiling_trace_frame_limit
                ):
                    if frame_id is not None:
                        self._trace_frame_ids.add(frame_id)
                    self._trace_events.append(
                        {
                            "name": name,
                            "ph": "B",
                            "ts": (active.started_ns - self._started_ns) / 1000.0,
                            "pid": os.getpid(),
                            "tid": threading.get_ident(),
                            "args": {"task_id": task_id, "frame_id": frame_id},
                        }
                    )
        return active

    def _exit_stage(
        self,
        active: _ActiveStage,
        *,
        frame_id: str | None,
        frame_count: int,
        batch_count: int,
        error: bool,
        cancelled: bool,
    ) -> None:
        ended_ns = perf_counter_ns()
        cpu_ended_ns = process_time_ns()
        duration = max(0, ended_ns - active.started_ns)
        stack = self._stack()
        if stack and stack[-1] is active:
            stack.pop()
        elif active in stack:
            stack.remove(active)
        if stack:
            stack[-1].child_ns += duration
        with self._lock:
            aggregate = self._stages.setdefault(active.name, StageAggregate(active.name))
            aggregate.add(
                duration,
                max(0, duration - active.child_ns),
                max(0, cpu_ended_ns - active.cpu_started_ns),
                frame_count=frame_count,
                batch_count=batch_count,
                error=error,
                cancelled=cancelled,
            )
            self._current_stage = stack[-1].name if stack else None
            if frame_id and duration / 1_000_000.0 >= self.config.profiling_slow_frame_threshold_ms:
                item = (duration, str(frame_id), active.name)
                if len(self._slow_frames) < _SLOW_FRAME_LIMIT:
                    heapq.heappush(self._slow_frames, item)
                elif item > self._slow_frames[0]:
                    heapq.heapreplace(self._slow_frames, item)
            if self.config.profiling_mode is ProfilingMode.TRACE and len(self._trace_events) < _TRACE_EVENT_LIMIT:
                self._trace_events.append(
                    {
                        "name": active.name,
                        "ph": "E",
                        "ts": (ended_ns - self._started_ns) / 1000.0,
                        "pid": os.getpid(),
                        "tid": threading.get_ident(),
                    }
                )

    def increment(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[str(name)] = self._counters.get(str(name), 0) + int(amount)

    def set_counter(self, name: str, value: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[str(name)] = int(value)

    def record_duration(
        self,
        name: str,
        duration_ns: int,
        *,
        frame_count: int = 0,
        batch_count: int = 0,
        error: bool = False,
        cancelled: bool = False,
    ) -> None:
        """Record work timed by an event-loop callback outside an active stage."""

        if not self.enabled:
            return
        measured_ns = max(0, int(duration_ns))
        with self._lock:
            self._stages.setdefault(str(name), StageAggregate(str(name))).add(
                measured_ns,
                measured_ns,
                0,
                frame_count=frame_count,
                batch_count=batch_count,
                error=error,
                cancelled=cancelled,
            )

    def merge_worker(self, packet: WorkerProfilePacket) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._worker_packets[packet.worker_id] = packet
            for stage in packet.stages:
                self._stages.setdefault(stage.name, StageAggregate(stage.name)).merge(stage)
            for name, value in packet.counters:
                self._counters[name] = self._counters.get(name, 0) + int(value)

    def worker_packet(self, worker_id: str, *, processed_frames: int, processed_batches: int) -> WorkerProfilePacket:
        with self._lock:
            return WorkerProfilePacket(
                worker_id=str(worker_id),
                pid=os.getpid(),
                stages=tuple(self._stages.values()),
                counters=tuple(sorted(self._counters.items())),
                processed_frames=max(0, int(processed_frames)),
                processed_batches=max(0, int(processed_batches)),
            )

    def snapshot(self) -> ProfileSnapshot:
        with self._lock:
            elapsed_ns = max(0, perf_counter_ns() - self._started_ns)
            stages = tuple(
                stage.to_payload(elapsed_ns)
                for stage in sorted(self._stages.values(), key=lambda item: item.self_ns, reverse=True)
            )
            slow_frames = tuple(
                {"frame_id": frame_id, "stage": stage, "duration_ms": duration / 1_000_000.0}
                for duration, frame_id, stage in sorted(self._slow_frames, reverse=True)
            )
            workers = tuple(
                {
                    "worker_id": packet.worker_id,
                    "pid": packet.pid,
                    "processed_frames": packet.processed_frames,
                    "processed_batches": packet.processed_batches,
                }
                for packet in sorted(self._worker_packets.values(), key=lambda item: item.worker_id)
            )
            return ProfileSnapshot(
                run_id=self.run_id,
                analysis_type=self.analysis_type,
                mode=self.config.profiling_mode,
                started_at=self.started_at,
                elapsed_ms=elapsed_ns / 1_000_000.0,
                current_stage=self._current_stage,
                processed_frames=int(self._counters.get("frames.processed", 0)),
                stages=stages,
                counters=dict(self._counters),
                slow_frames=slow_frames,
                workers=workers,
                environment=_environment_payload(self.config),
                trace_events=tuple(self._trace_events),
            )


_active_profiler = threading.local()


def current_profiler() -> ProfilerRun | None:
    return getattr(_active_profiler, "value", None)


@contextmanager
def activate_profiler(profiler: ProfilerRun | None) -> Iterator[ProfilerRun | None]:
    previous = current_profiler()
    _active_profiler.value = profiler
    try:
        yield profiler
    finally:
        _active_profiler.value = previous


def profile_stage(name: str, *, frame_id: str | None = None, frame_count: int = 0) -> AbstractContextManager[None]:
    profiler = current_profiler()
    if profiler is None or not profiler.enabled:
        return _NULL_STAGE
    return profiler.stage(name, frame_id=frame_id, frame_count=frame_count)


class _NullStage(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


_NULL_STAGE = _NullStage()


def export_profile(snapshot: ProfileSnapshot, directory: Path, config: PerformanceConfig) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = f"karakal_profile_{snapshot.analysis_type}_{snapshot.processed_frames}_{stamp}_{snapshot.run_id[:8]}"
    exported: dict[str, Path] = {}
    if config.profiling_export_json:
        path = directory / f"{stem}.json"
        _atomic_text(path, json.dumps(snapshot.to_payload(), ensure_ascii=False, indent=2))
        exported["json"] = path
    if config.profiling_export_csv:
        path = directory / f"{stem}.csv"
        _atomic_csv(path, snapshot.stages)
        exported["csv"] = path
    if config.profiling_export_markdown:
        path = directory / f"{stem}.md"
        _atomic_text(path, _markdown(snapshot))
        exported["markdown"] = path
    if snapshot.mode is ProfilingMode.TRACE or config.profiling_trace_enabled:
        path = directory / f"{stem}.trace.json"
        _atomic_text(path, json.dumps({"traceEvents": list(snapshot.trace_events)}, ensure_ascii=False))
        exported["trace"] = path
    _trim_profile_history(directory, config.profiling_keep_last_runs)
    return exported


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(rows[0].keys()) if rows else ["name", "calls", "total_ms", "self_ms"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _markdown(snapshot: ProfileSnapshot) -> str:
    lines = [
        f"# Karakal profile `{snapshot.run_id}`",
        "",
        f"- Analysis: `{snapshot.analysis_type}`",
        f"- Mode: `{snapshot.mode.value}`",
        f"- Elapsed: `{snapshot.elapsed_ms:.3f} ms`",
        f"- Frames: `{snapshot.processed_frames}`",
        "",
        "| Stage | Calls | Total ms | Self ms | Avg ms | P95 ms | Share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in snapshot.stages:
        lines.append(
            f"| {row['name']} | {row['calls']} | {float(row['total_ms']):.3f} | "
            f"{float(row['self_ms']):.3f} | {float(row['average_ms'] or 0.0):.3f} | "
            f"{float(row['p95_ms'] or 0.0):.3f} | {100.0 * float(row['share']):.2f}% |"
        )
    return "\n".join(lines) + "\n"


def _trim_profile_history(directory: Path, keep_runs: int) -> None:
    groups: dict[str, list[Path]] = {}
    for path in directory.glob("karakal_profile_*"):
        stem = path.name.split(".trace.json", 1)[0].rsplit(".", 1)[0]
        groups.setdefault(stem, []).append(path)
    ordered = sorted(groups.values(), key=lambda paths: max(item.stat().st_mtime_ns for item in paths), reverse=True)
    for paths in ordered[max(1, int(keep_runs)) :]:
        for path in paths:
            path.unlink(missing_ok=True)


def _environment_payload(config: PerformanceConfig) -> dict[str, object]:
    try:
        import cv2

        opencv_version: str | None = str(cv2.__version__)
    except ImportError:
        opencv_version = None
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
    except ImportError:
        PYQT_VERSION_STR = None
        QT_VERSION_STR = None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": opencv_version,
        "pyqt": PYQT_VERSION_STR,
        "qt": QT_VERSION_STR,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "workers": config.cpu_workers,
        "batch_size": config.batch_size,
        "profiling_mode": config.profiling_mode.value,
        "peak_ram_bytes": None,
        "cpu_utilization": None,
    }


__all__ = [
    "ProfileSnapshot",
    "ProfilerRun",
    "StageAggregate",
    "WorkerProfilePacket",
    "activate_profiler",
    "current_profiler",
    "export_profile",
    "profile_stage",
]
