"""Central profiling controls and file output for Contour.

Environment switches:
  CONTOUR_PROFILE=0 disables all contour profiling.
  CONTOUR_PROFILE=1 enables profiling unless a narrower switch overrides it.
  CONTOUR_PROFILE_<KIND>=0/1 controls a specific profiler.

The bundled ``contour.ini`` is the primary source of defaults. Environment
variables remain supported as deployment/debug overrides.
"""

from __future__ import annotations

import cProfile
import logging
import os
import re
from datetime import datetime
from logging import FileHandler
from pathlib import Path
from threading import Lock

from .runtime_config import config_bool, config_int, load_runtime_config, profiling_log_path

PROFILE_ENV_TRUE = {"1", "true", "yes", "on"}
PROFILE_ENV_FALSE = {"0", "false", "no", "off"}

DEFAULT_FRAME_SWITCH_TOP_LINES = 80
DEFAULT_STARTUP_TOP_LINES = 80
DEFAULT_PROCESSING_TOP_LINES = 25
DEFAULT_THUMBNAIL_TOP_LINES = 25
DEFAULT_VERTEX_MOVE_TOP_LINES = 40
DEFAULT_MOVE_VERTEX_TOOL_TOP_LINES = 40
DEFAULT_DELETE_AREA_TOP_LINES = 40
DEFAULT_CONTACT_PLACEMENT_TOP_LINES = 40
DEFAULT_CONTACT_MULTI_SELECTION_TOP_LINES = 40
DEFAULT_CONTACT_DELETION_TOP_LINES = 40
DEFAULT_CONTACT_COPY_TOP_LINES = 40
DEFAULT_CONTACT_PASTE_TOP_LINES = 40
DEFAULT_CONTACT_UNDO_TOP_LINES = 40
DEFAULT_CONTACT_REDO_TOP_LINES = 40
DEFAULT_CONTACT_DRAG_TOP_LINES = 40
DEFAULT_SCENE_ZOOM_TOP_LINES = 40
DEFAULT_IMAGE_RECOGNITION_TOP_LINES = 40
DEFAULT_POLYGON_CHANGE_TOP_LINES = 40
DEFAULT_FRAME_SWITCH_IDLE_POLLS = 300
MAX_PROFILE_LOG_FILES = 500

_PROFILE_LOGGER_NAME = "contour.profiling.output"
_PROFILE_LOG_LOCK = Lock()
_PROFILE_LOG_HANDLER: FileHandler | None = None
_LOGGER = logging.getLogger(__name__)


def _new_profile_log_path(configured_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return configured_path.with_name(
        f"{configured_path.stem}-{timestamp}-p{os.getpid()}{configured_path.suffix}"
    )


def _profile_log_files(configured_path: Path) -> list[Path]:
    generated = list(configured_path.parent.glob(f"{configured_path.stem}-*-p*{configured_path.suffix}"))
    if configured_path.is_file():
        generated.append(configured_path)
    return generated


def _prune_profile_logs(configured_path: Path) -> None:
    candidates = _profile_log_files(configured_path)
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    for stale_path in candidates[MAX_PROFILE_LOG_FILES:]:
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            _LOGGER.warning("Cannot remove old Contour profiling log: %s", stale_path, exc_info=True)


def _env_flag(name: str) -> bool | None:
    value = str(os.environ.get(name, "")).strip().lower()
    if value in PROFILE_ENV_TRUE:
        return True
    if value in PROFILE_ENV_FALSE:
        return False
    return None


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = str(os.environ.get(name, "")).strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(minimum, value)


# CPython 3.13+ reimplemented the legacy profiler slot on top of
# sys.monitoring and reports a conflict with the tool id
# ("tool 2 is already in use"); older builds use the legacy wording.
_PROFILER_CONFLICT_LEGACY_MESSAGE = "Another profiling tool is already active"
_PROFILER_CONFLICT_TOOL_IN_USE = re.compile(r"\btool\s+\d+\s+is already in use\b")


def profiler_enable_conflict(exc: BaseException) -> bool:
    message = str(exc)
    return (
        _PROFILER_CONFLICT_LEGACY_MESSAGE in message
        or _PROFILER_CONFLICT_TOOL_IN_USE.search(message) is not None
    )


# Since CPython 3.12 cProfile runs on sys.monitoring: only one profiler can be enabled per
# process, and it records every thread. Separate per-session/per-thread profilers therefore
# evict each other (the loser silently gets no data). SharedProfile instead keeps one host
# profiler enabled while any session needs it and gives each session the stats delta between
# its own enable() and disable(). Work in any thread during that window is included; a call
# is counted when it returns, so a call still running at disable() is not in the window yet.
_HOST_LOCK = Lock()
_HOST_PROFILER: cProfile.Profile | None = None
_HOST_LEASES = 0

# code -> [callcount, reccallcount, inlinetime, totaltime, {subcode: [cc, rcc, inline, total]}]
_StatsSnapshot = dict


def _host_snapshot_locked() -> _StatsSnapshot:
    assert _HOST_PROFILER is not None
    snapshot: _StatsSnapshot = {}
    for entry in _HOST_PROFILER.getstats():
        calls = {
            sub.code: [sub.callcount, sub.reccallcount, sub.inlinetime, sub.totaltime]
            for sub in (entry.calls or ())
        }
        snapshot[entry.code] = [entry.callcount, entry.reccallcount, entry.inlinetime, entry.totaltime, calls]
    return snapshot


def _host_acquire() -> _StatsSnapshot:
    global _HOST_PROFILER, _HOST_LEASES
    with _HOST_LOCK:
        if _HOST_LEASES == 0:
            host = cProfile.Profile()
            host.enable()  # ValueError when a profiler outside SharedProfile holds the slot
            _HOST_PROFILER = host
        _HOST_LEASES += 1
        return _host_snapshot_locked()


def _host_release() -> _StatsSnapshot:
    global _HOST_PROFILER, _HOST_LEASES
    with _HOST_LOCK:
        snapshot = _host_snapshot_locked()
        _HOST_LEASES -= 1
        if _HOST_LEASES == 0 and _HOST_PROFILER is not None:
            try:
                _HOST_PROFILER.disable()
            except ValueError:
                pass
            _HOST_PROFILER = None
        return snapshot


def _host_peek() -> _StatsSnapshot:
    with _HOST_LOCK:
        return _host_snapshot_locked() if _HOST_PROFILER is not None else {}


def _accumulate_delta(target: _StatsSnapshot, start: _StatsSnapshot, end: _StatsSnapshot) -> None:
    for code, (nc, rc, inline, total, calls) in end.items():
        base = start.get(code)
        if base is not None:
            nc, rc, inline, total = nc - base[0], rc - base[1], inline - base[2], total - base[3]
        if nc <= 0:
            continue
        acc = target.setdefault(code, [0, 0, 0.0, 0.0, {}])
        acc[0] += nc
        acc[1] += rc
        acc[2] += inline
        acc[3] += total
        base_calls = base[4] if base is not None else {}
        for sub_code, (snc, src, sinline, stotal) in calls.items():
            sub_base = base_calls.get(sub_code)
            if sub_base is not None:
                snc, src, sinline, stotal = (
                    snc - sub_base[0],
                    src - sub_base[1],
                    sinline - sub_base[2],
                    stotal - sub_base[3],
                )
            if snc <= 0:
                continue
            sub_acc = acc[4].setdefault(sub_code, [0, 0, 0.0, 0.0])
            sub_acc[0] += snc
            sub_acc[1] += src
            sub_acc[2] += sinline
            sub_acc[3] += stotal


class _Entry:
    __slots__ = ("callcount", "calls", "code", "inlinetime", "reccallcount", "totaltime")

    def __init__(self, code, callcount, reccallcount, inlinetime, totaltime, calls) -> None:
        self.code = code
        self.callcount = callcount
        self.reccallcount = reccallcount
        self.inlinetime = inlinetime
        self.totaltime = totaltime
        self.calls = calls


class SharedProfile(cProfile.Profile):
    """A session's window onto the process-wide host profiler; a drop-in for cProfile.Profile."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._window_start: _StatsSnapshot | None = None
        self._collected: _StatsSnapshot = {}

    @property
    def window_active(self) -> bool:
        return self._window_start is not None

    def enable(self, *args, **kwargs) -> None:
        if self._window_start is None:
            self._window_start = _host_acquire()

    def disable(self) -> None:
        start = self._window_start
        if start is None:
            return
        self._window_start = None
        _accumulate_delta(self._collected, start, _host_release())

    def clear(self) -> None:
        self._collected = {}
        if self._window_start is not None:
            self._window_start = _host_peek()

    def getstats(self) -> list:
        collected: _StatsSnapshot = {}
        _accumulate_delta(collected, {}, self._collected)
        if self._window_start is not None:
            _accumulate_delta(collected, self._window_start, _host_peek())
        return [
            _Entry(
                code,
                nc,
                rc,
                inline,
                total,
                [_Entry(sub, *values, None) for sub, values in calls.items()] or None,
            )
            for code, (nc, rc, inline, total, calls) in collected.items()
        ]


def try_enable_profiler(profiler: cProfile.Profile) -> bool:
    try:
        profiler.enable()
        return True
    except ValueError as exc:
        if profiler_enable_conflict(exc):
            return False
        raise


def try_disable_profiler(profiler: cProfile.Profile) -> None:
    try:
        profiler.disable()
    except ValueError:
        pass


def _master_profile_flag() -> bool | None:
    for name in ("CONTOUR_PROFILE", "CONTOUR_PROFILING", "CONTOUR_PROFILE_ALL"):
        value = _env_flag(name)
        if value is not None:
            return value
    return None


def _ini_flag(kind: str) -> bool | None:
    parser = load_runtime_config()
    if parser.has_option("profiling", kind):
        try:
            return parser.getboolean("profiling", kind)
        except ValueError:
            return None
    return None


def profiling_enabled(kind: str, *, default: bool = False, legacy_env: tuple[str, ...] = ()) -> bool:
    env_name = f"CONTOUR_PROFILE_{kind.upper()}"
    explicit = _env_flag(env_name)
    if explicit is not None:
        return explicit
    for legacy_name in legacy_env:
        legacy = _env_flag(legacy_name)
        if legacy is not None:
            return legacy
    master = _master_profile_flag()
    if master is not None:
        return master
    specific_ini = _ini_flag(kind)
    if specific_ini is not None:
        return specific_ini
    if load_runtime_config().has_option("profiling", "enabled"):
        return config_bool("profiling", "enabled", default)
    return default


def profiling_top_lines(kind: str, default: int) -> int:
    specific = _env_int(f"CONTOUR_PROFILE_{kind.upper()}_TOP", 0, minimum=0)
    if specific > 0:
        return specific
    common_env = _env_int("CONTOUR_PROFILE_TOP", 0, minimum=0)
    if common_env > 0:
        return common_env
    return config_int(
        "profiling",
        f"{kind}_top_lines",
        config_int("profiling", "top_lines", default, minimum=1),
        minimum=1,
    )


def write_profile_report(*messages: object) -> None:
    """Append one profiling report to the configured rotating UTF-8 log."""

    global _PROFILE_LOG_HANDLER
    text = "\n".join(str(message).rstrip() for message in messages if message is not None)
    if not text:
        return
    with _PROFILE_LOG_LOCK:
        logger = logging.getLogger(_PROFILE_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if _PROFILE_LOG_HANDLER is None:
            configured_path = profiling_log_path()
            path = _new_profile_log_path(configured_path)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                _PROFILE_LOG_HANDLER = FileHandler(path, mode="x", encoding="utf-8")
                _prune_profile_logs(configured_path)
            except OSError:
                _LOGGER.exception("Cannot open Contour profiling log: %s", path)
                return
            _PROFILE_LOG_HANDLER.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(_PROFILE_LOG_HANDLER)
        logger.info(text)


def reset_profile_output() -> None:
    """Close the lazy profiling log handler so configuration can be reloaded."""

    global _PROFILE_LOG_HANDLER
    with _PROFILE_LOG_LOCK:
        if _PROFILE_LOG_HANDLER is None:
            return
        logger = logging.getLogger(_PROFILE_LOGGER_NAME)
        logger.removeHandler(_PROFILE_LOG_HANDLER)
        _PROFILE_LOG_HANDLER.close()
        _PROFILE_LOG_HANDLER = None


def frame_switch_profiling_enabled() -> bool:
    return profiling_enabled(
        "frame_switch",
        legacy_env=("CONTOUR_PROFILE_FRAME_OPEN", "CONTOUR_PROFILE_CIF_OPEN"),
    )


def startup_profiling_enabled() -> bool:
    return profiling_enabled("startup")


def processing_profiling_enabled() -> bool:
    return profiling_enabled("processing")


def thumbnail_profiling_enabled() -> bool:
    return profiling_enabled("thumbnail")


def thumbnail_full_function_usage_enabled() -> bool:
    env_value = _env_flag("CONTOUR_PROFILE_THUMBNAIL_FULL")
    if env_value is not None:
        return env_value
    return config_bool("profiling", "thumbnail_full_functions", False)


def vertex_move_profiling_enabled() -> bool:
    return profiling_enabled("vertex_move")


def move_vertex_tool_profiling_enabled() -> bool:
    return profiling_enabled(
        "move_vertex_tool",
        legacy_env=("CONTOUR_PROFILE_VERTEX_TOOL",),
    )


def delete_area_profiling_enabled() -> bool:
    return profiling_enabled(
        "delete_area",
    )


def contact_placement_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_placement",
    )


def contact_multi_selection_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_multi_selection",
    )


def contact_deletion_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_deletion",
    )


def contact_copy_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_copy",
    )


def contact_paste_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_paste",
    )


def contact_undo_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_undo",
    )


def contact_redo_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_redo",
    )


def contact_drag_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_drag",
    )


def scene_zoom_profiling_enabled() -> bool:
    return profiling_enabled(
        "scene_zoom",
    )


def image_recognition_profiling_enabled() -> bool:
    return profiling_enabled(
        "image_recognition",
    )


def filter_application_profiling_enabled() -> bool:
    return profiling_enabled(
        "filter_application",
    )


def polygon_change_profiling_enabled() -> bool:
    return profiling_enabled(
        "polygon_change",
    )


def frame_switch_top_lines() -> int:
    return profiling_top_lines("frame_switch", DEFAULT_FRAME_SWITCH_TOP_LINES)


def startup_top_lines() -> int:
    return profiling_top_lines("startup", DEFAULT_STARTUP_TOP_LINES)


def processing_top_lines() -> int:
    return profiling_top_lines("processing", DEFAULT_PROCESSING_TOP_LINES)


def thumbnail_top_lines() -> int:
    return profiling_top_lines("thumbnail", DEFAULT_THUMBNAIL_TOP_LINES)


def vertex_move_top_lines() -> int:
    return profiling_top_lines("vertex_move", DEFAULT_VERTEX_MOVE_TOP_LINES)


def move_vertex_tool_top_lines() -> int:
    return profiling_top_lines("move_vertex_tool", DEFAULT_MOVE_VERTEX_TOOL_TOP_LINES)


def delete_area_top_lines() -> int:
    return profiling_top_lines("delete_area", DEFAULT_DELETE_AREA_TOP_LINES)


def contact_placement_top_lines() -> int:
    return profiling_top_lines("contact_placement", DEFAULT_CONTACT_PLACEMENT_TOP_LINES)


def contact_multi_selection_top_lines() -> int:
    return profiling_top_lines(
        "contact_multi_selection",
        DEFAULT_CONTACT_MULTI_SELECTION_TOP_LINES,
    )


def contact_deletion_top_lines() -> int:
    return profiling_top_lines("contact_deletion", DEFAULT_CONTACT_DELETION_TOP_LINES)


def contact_copy_top_lines() -> int:
    return profiling_top_lines("contact_copy", DEFAULT_CONTACT_COPY_TOP_LINES)


def contact_paste_top_lines() -> int:
    return profiling_top_lines("contact_paste", DEFAULT_CONTACT_PASTE_TOP_LINES)


def contact_undo_top_lines() -> int:
    return profiling_top_lines("contact_undo", DEFAULT_CONTACT_UNDO_TOP_LINES)


def contact_redo_top_lines() -> int:
    return profiling_top_lines("contact_redo", DEFAULT_CONTACT_REDO_TOP_LINES)


def contact_drag_top_lines() -> int:
    return profiling_top_lines("contact_drag", DEFAULT_CONTACT_DRAG_TOP_LINES)


def scene_zoom_top_lines() -> int:
    return profiling_top_lines("scene_zoom", DEFAULT_SCENE_ZOOM_TOP_LINES)


def image_recognition_top_lines() -> int:
    return profiling_top_lines(
        "image_recognition",
        DEFAULT_IMAGE_RECOGNITION_TOP_LINES,
    )


def polygon_change_top_lines() -> int:
    return profiling_top_lines("polygon_change", DEFAULT_POLYGON_CHANGE_TOP_LINES)


def frame_switch_idle_polls() -> int:
    env_value = _env_int("CONTOUR_PROFILE_FRAME_SWITCH_IDLE_POLLS", 0, minimum=0)
    if env_value > 0:
        return env_value
    return config_int(
        "profiling",
        "frame_switch_idle_polls",
        DEFAULT_FRAME_SWITCH_IDLE_POLLS,
        minimum=1,
    )
