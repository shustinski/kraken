"""Opt-in profiling reports for the Kraken desktop and frame matrix."""

from __future__ import annotations

import configparser
import cProfile
import io
import os
import profile as thread_profile
import pstats
import re
import sys
from contextlib import contextmanager
from concurrent.futures import Future
from dataclasses import dataclass
from logging import INFO, LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Callable, Iterator, TypeVar

from kraken_core.runtime import package_runtime_root

_KINDS = (
    "main_window",
    "matrix_viewport",
    "matrix_thumbnail",
    "matrix_ui",
    "matrix_thumbnails_total",
    "layer_selection",
)
_DEFAULT_INI = """; Kraken profiling. Restart Kraken after editing.
[profiling]
enabled = false
; Empty values inherit enabled. Set true or false to override individually.
; Main window: from completed sign-in through the first GUI paint and event-loop callback.
main_window =
matrix_viewport =
; Choose one thumbnail mode: individual or total. Both cannot be true.
matrix_thumbnail = false
matrix_ui =
; Wall time for all thumbnails requested by one visible viewport.
matrix_thumbnails_total =
; Layer area selection: from the mouse press through the redraw after release.
layer_selection =
top_lines = 25
log_file = profiling.log
"""


@dataclass(frozen=True)
class _ProfileConfig:
    flags: dict[str, bool]
    top_lines: int
    log_path: Path


_config: _ProfileConfig | None = None
_handler: RotatingFileHandler | None = None
_handler_lock = Lock()
_Result = TypeVar("_Result")


class _WorkerCallProfile(thread_profile.Profile):
    """Profile one worker call without aborting it.

    The stdlib profiler asserts when a C return does not match the parallel
    stack. Socket reads from several thumbnail workers hit that path and the
    assertion cancels the download.
    """

    def trace_dispatch_return(self, frame, t):
        current = self.cur
        if current is not None and frame is not current[-2]:
            parent = getattr(current[-2], "f_back", None)
            if frame is not parent:
                return 1
        return thread_profile.Profile.trace_dispatch_return(self, frame, t)


_WorkerCallProfile.dispatch = dict(thread_profile.Profile.dispatch)
_WorkerCallProfile.dispatch["return"] = _WorkerCallProfile.trace_dispatch_return
_WorkerCallProfile.dispatch["c_return"] = _WorkerCallProfile.trace_dispatch_return
_WorkerCallProfile.dispatch["c_exception"] = _WorkerCallProfile.trace_dispatch_return


def project_directory() -> Path:
    """Use the repository root in development and the executable folder in a build."""
    if bool(getattr(sys, "frozen", False)):
        return Path(sys.executable).resolve().parent
    return package_runtime_root(__file__)


def configure(project_dir: str | Path) -> Path:
    """Create the editable INI on first start and load its settings for this process."""
    global _config, _handler
    directory = Path(project_dir).expanduser().resolve()
    explicit = os.environ.get("KRAKEN_CONFIG", "").strip()
    ini_path = Path(explicit).expanduser().resolve() if explicit else directory / "kraken.ini"
    if not ini_path.exists():
        if explicit:
            raise FileNotFoundError(f"Kraken config does not exist: {ini_path}")
        ini_path.parent.mkdir(parents=True, exist_ok=True)
        ini_path.write_text(_DEFAULT_INI, encoding="utf-8")

    parser = configparser.ConfigParser(interpolation=None)
    with ini_path.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    section = "profiling"
    master = parser.getboolean(section, "enabled", fallback=False)
    flags: dict[str, bool] = {}
    for kind in _KINDS:
        raw = parser.get(section, kind, fallback="").strip()
        try:
            flags[kind] = parser.getboolean(section, kind) if raw else master
        except ValueError as exc:
            raise ValueError(f"invalid Kraken profiling setting {kind}: {raw!r}") from exc
    if flags["matrix_thumbnail"] and flags["matrix_thumbnails_total"]:
        raise ValueError("Kraken profiling: matrix_thumbnail and matrix_thumbnails_total cannot both be true")
    top_lines = parser.getint(section, "top_lines", fallback=25)
    if not 1 <= top_lines <= 200:
        raise ValueError("Kraken profiling top_lines must be between 1 and 200")
    log_file = parser.get(section, "log_file", fallback="profiling.log").strip()
    if not log_file:
        raise ValueError("Kraken profiling log_file must not be empty")
    log_path = Path(log_file).expanduser()
    if not log_path.is_absolute():
        log_path = ini_path.parent / log_path
    with _handler_lock:
        if _handler is not None:
            _handler.close()
            _handler = None
    _config = _ProfileConfig(flags, top_lines, log_path)
    return ini_path


def enabled(kind: str) -> bool:
    config = _config
    return config is not None and config.flags.get(kind, False)


@dataclass
class ProfileSession:
    kind: str
    config: _ProfileConfig
    profiler: cProfile.Profile
    started: float
    active: bool
    finished: bool = False

    def finish(self, *, ready: bool | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        if self.active:
            self.profiler.disable()
        report = io.StringIO()
        state = "" if ready is None else f" ready={'yes' if ready else 'no'}"
        report.write(f"[{self.kind}] elapsed_ms={(perf_counter() - self.started) * 1000:.3f}{state}\n")
        if self.active:
            pstats.Stats(self.profiler, stream=report).strip_dirs().sort_stats("cumulative").print_stats(
                self.config.top_lines
            )
        else:
            report.write("cprofile_skipped=yes reason=another_profiler_active\n")
        _write_report(self.config.log_path, report.getvalue())


def begin_region(kind: str) -> ProfileSession | None:
    """Start profiling now; the caller ends it after synchronous or queued work."""
    config = _config
    if config is None or not config.flags.get(kind, False):
        return None
    profiler = cProfile.Profile()
    started = perf_counter()
    try:
        profiler.enable()
        active = True
    except ValueError as exc:
        if not _is_profiler_conflict(exc):
            raise
        active = False
    return ProfileSession(kind, config, profiler, started, active)


@contextmanager
def region(kind: str) -> Iterator[None]:
    """Profile one synchronous operation on its calling thread."""
    session = begin_region(kind)
    try:
        yield
    finally:
        if session is not None:
            session.finish()


class ThumbnailBatchProfile:
    """Measure the complete load and sample worker calls without serializing them."""

    def __init__(self, config: _ProfileConfig) -> None:
        self.config = config
        self.started = perf_counter()
        self.finished = False
        self._workers: pstats.Stats | None = None
        self._lock = Lock()
        self._profiled_calls = 0
        self._pending: set[Future[object]] = set()

    def track(self, future: Future[object]) -> None:
        with self._lock:
            if future in self._pending:
                return
            self._pending.add(future)
        future.add_done_callback(self._completed)

    def _completed(self, future: Future[object]) -> None:
        with self._lock:
            self._pending.discard(future)

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def profile_call(self, call: Callable[[], _Result]) -> _Result:
        profiler = _WorkerCallProfile(timer=perf_counter)
        try:
            return profiler.runcall(call)
        finally:
            self._record_worker_profile(profiler)

    def _record_worker_profile(self, profiler: _WorkerCallProfile) -> None:
        try:
            stats = pstats.Stats(profiler, stream=io.StringIO())
        except (AssertionError, TypeError, ValueError):
            return
        with self._lock:
            if self._workers is None:
                self._workers = stats
            else:
                self._workers.add(stats)
            self._profiled_calls += 1

    def cancel(self) -> None:
        self.finished = True

    def finish(
        self,
        *,
        requested: int,
        completed: int,
        failed: int,
        superseded: int,
        error_types: dict[str, int],
    ) -> None:
        if self.finished:
            return
        self.finished = True
        report = io.StringIO()
        report.write(
            f"[matrix_thumbnails_total] elapsed_ms={(perf_counter() - self.started) * 1000:.3f} "
            f"requested={requested} completed={completed} failed={failed} superseded={superseded} "
            f"status={'failed' if failed else 'complete'}\n"
        )
        if error_types:
            report.write(
                "error_types=" + ",".join(f"{kind}:{count}" for kind, count in sorted(error_types.items())) + "\n"
            )
        report.write(f"Worker calls profiled={self._profiled_calls} (summed across threads; excludes GUI):\n")
        if self._workers is not None:
            self._workers.stream = report
            self._workers.strip_dirs().sort_stats("cumulative").print_stats(self.config.top_lines)
        _write_report(self.config.log_path, report.getvalue())


def begin_thumbnail_batch() -> ThumbnailBatchProfile | None:
    config = _config
    return (
        ThumbnailBatchProfile(config)
        if config is not None and config.flags.get("matrix_thumbnails_total", False)
        else None
    )


def _is_profiler_conflict(exc: ValueError) -> bool:
    message = str(exc)
    return (
        "Another profiling tool is already active" in message
        or re.search(r"\btool\s+\d+\s+is already in use\b", message) is not None
    )


def _write_report(path: Path, report: str) -> None:
    global _handler
    # Logging handlers serialize concurrent writes from viewport and thumbnail workers.
    with _handler_lock:
        if _handler is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            _handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=3, encoding="utf-8", delay=True)
        record = LogRecord("kraken.matrix.profiling", INFO, "", 0, "%s", (report,), None)
        _handler.handle(record)
