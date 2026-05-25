"""Central profiling controls for Contour.

Environment switches:
  CONTOUR_PROFILE=0 disables all contour profiling.
  CONTOUR_PROFILE=1 enables profiling unless a narrower switch overrides it.
  CONTOUR_PROFILE_<KIND>=0/1 controls a specific profiler.
"""

from __future__ import annotations

import cProfile
import os

PROFILE_ENV_TRUE = {"1", "true", "yes", "on"}
PROFILE_ENV_FALSE = {"0", "false", "no", "off"}

DEFAULT_FRAME_SWITCH_ENABLED = False
DEFAULT_PROCESSING_ENABLED = False
DEFAULT_THUMBNAIL_ENABLED = False
DEFAULT_VERTEX_MOVE_ENABLED = False

DEFAULT_FRAME_SWITCH_TOP_LINES = 80
DEFAULT_PROCESSING_TOP_LINES = 25
DEFAULT_THUMBNAIL_TOP_LINES = 25
DEFAULT_VERTEX_MOVE_TOP_LINES = 40
DEFAULT_FRAME_SWITCH_IDLE_POLLS = 300


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


def profiler_enable_conflict(exc: BaseException) -> bool:
    return "Another profiling tool is already active" in str(exc)


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


def profiling_enabled(kind: str, *, default: bool, legacy_env: tuple[str, ...] = ()) -> bool:
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
    return default


def profiling_top_lines(kind: str, default: int) -> int:
    specific = _env_int(f"CONTOUR_PROFILE_{kind.upper()}_TOP", 0, minimum=0)
    if specific > 0:
        return specific
    return _env_int("CONTOUR_PROFILE_TOP", default)


def frame_switch_profiling_enabled() -> bool:
    return profiling_enabled(
        "frame_switch",
        default=DEFAULT_FRAME_SWITCH_ENABLED,
        legacy_env=("CONTOUR_PROFILE_FRAME_OPEN", "CONTOUR_PROFILE_CIF_OPEN"),
    )


def processing_profiling_enabled() -> bool:
    return profiling_enabled("processing", default=DEFAULT_PROCESSING_ENABLED)


def thumbnail_profiling_enabled() -> bool:
    return profiling_enabled("thumbnail", default=DEFAULT_THUMBNAIL_ENABLED)


def thumbnail_full_function_usage_enabled() -> bool:
    return bool(_env_flag("CONTOUR_PROFILE_THUMBNAIL_FULL"))


def vertex_move_profiling_enabled() -> bool:
    return profiling_enabled("vertex_move", default=DEFAULT_VERTEX_MOVE_ENABLED)


def frame_switch_top_lines() -> int:
    return profiling_top_lines("frame_switch", DEFAULT_FRAME_SWITCH_TOP_LINES)


def processing_top_lines() -> int:
    return profiling_top_lines("processing", DEFAULT_PROCESSING_TOP_LINES)


def thumbnail_top_lines() -> int:
    return profiling_top_lines("thumbnail", DEFAULT_THUMBNAIL_TOP_LINES)


def vertex_move_top_lines() -> int:
    return profiling_top_lines("vertex_move", DEFAULT_VERTEX_MOVE_TOP_LINES)


def frame_switch_idle_polls() -> int:
    return _env_int("CONTOUR_PROFILE_FRAME_SWITCH_IDLE_POLLS", DEFAULT_FRAME_SWITCH_IDLE_POLLS)
