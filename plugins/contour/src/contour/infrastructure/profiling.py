"""Central profiling controls for Contour.

Environment switches:
  CONTOUR_PROFILE=0 disables all contour profiling.
  CONTOUR_PROFILE=1 enables profiling unless a narrower switch overrides it.
  CONTOUR_PROFILE_<KIND>=0/1 controls a specific profiler.

The contact-placement profiler is normally toggled with the
CONTACT_PLACEMENT_PROFILING_ENABLED code variable below.
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
# Change this value to True to profile one contact placement end-to-end.
# CONTOUR_PROFILE_CONTACT_PLACEMENT=0/1 can still override it at runtime.
CONTACT_PLACEMENT_PROFILING_ENABLED = False
CONTACT_MULTI_SELECTION_PROFILING_ENABLED = False
CONTACT_DELETION_PROFILING_ENABLED = False
CONTACT_COPY_PROFILING_ENABLED = False
CONTACT_PASTE_PROFILING_ENABLED = False
CONTACT_UNDO_PROFILING_ENABLED = False
CONTACT_REDO_PROFILING_ENABLED = False
CONTACT_DRAG_PROFILING_ENABLED = False
SCENE_ZOOM_PROFILING_ENABLED = False
IMAGE_RECOGNITION_PROFILING_ENABLED = True

DEFAULT_FRAME_SWITCH_TOP_LINES = 80
DEFAULT_PROCESSING_TOP_LINES = 25
DEFAULT_THUMBNAIL_TOP_LINES = 25
DEFAULT_VERTEX_MOVE_TOP_LINES = 40
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


def contact_placement_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_placement",
        default=CONTACT_PLACEMENT_PROFILING_ENABLED,
    )


def contact_multi_selection_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_multi_selection",
        default=CONTACT_MULTI_SELECTION_PROFILING_ENABLED,
    )


def contact_deletion_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_deletion",
        default=CONTACT_DELETION_PROFILING_ENABLED,
    )


def contact_copy_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_copy",
        default=CONTACT_COPY_PROFILING_ENABLED,
    )


def contact_paste_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_paste",
        default=CONTACT_PASTE_PROFILING_ENABLED,
    )


def contact_undo_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_undo",
        default=CONTACT_UNDO_PROFILING_ENABLED,
    )


def contact_redo_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_redo",
        default=CONTACT_REDO_PROFILING_ENABLED,
    )


def contact_drag_profiling_enabled() -> bool:
    return profiling_enabled(
        "contact_drag",
        default=CONTACT_DRAG_PROFILING_ENABLED,
    )


def scene_zoom_profiling_enabled() -> bool:
    return profiling_enabled(
        "scene_zoom",
        default=SCENE_ZOOM_PROFILING_ENABLED,
    )


def image_recognition_profiling_enabled() -> bool:
    return profiling_enabled(
        "image_recognition",
        default=IMAGE_RECOGNITION_PROFILING_ENABLED,
    )


def frame_switch_top_lines() -> int:
    return profiling_top_lines("frame_switch", DEFAULT_FRAME_SWITCH_TOP_LINES)


def processing_top_lines() -> int:
    return profiling_top_lines("processing", DEFAULT_PROCESSING_TOP_LINES)


def thumbnail_top_lines() -> int:
    return profiling_top_lines("thumbnail", DEFAULT_THUMBNAIL_TOP_LINES)


def vertex_move_top_lines() -> int:
    return profiling_top_lines("vertex_move", DEFAULT_VERTEX_MOVE_TOP_LINES)


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


def frame_switch_idle_polls() -> int:
    return _env_int("CONTOUR_PROFILE_FRAME_SWITCH_IDLE_POLLS", DEFAULT_FRAME_SWITCH_IDLE_POLLS)
