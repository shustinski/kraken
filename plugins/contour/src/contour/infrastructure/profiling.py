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
from logging.handlers import RotatingFileHandler
from threading import Lock

from .runtime_config import config_bool, config_int, load_runtime_config, profiling_log_path

PROFILE_ENV_TRUE = {"1", "true", "yes", "on"}
PROFILE_ENV_FALSE = {"0", "false", "no", "off"}

DEFAULT_FRAME_SWITCH_TOP_LINES = 80
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

_PROFILE_LOGGER_NAME = "contour.profiling.output"
_PROFILE_LOG_LOCK = Lock()
_PROFILE_LOG_HANDLER: RotatingFileHandler | None = None
_LOGGER = logging.getLogger(__name__)


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
            path = profiling_log_path()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                _PROFILE_LOG_HANDLER = RotatingFileHandler(
                    path,
                    maxBytes=config_int("profiling", "max_log_bytes", 5 * 1024 * 1024, minimum=1024),
                    backupCount=config_int("profiling", "backup_count", 3, minimum=0),
                    encoding="utf-8",
                )
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
