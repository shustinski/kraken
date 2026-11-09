"""Central cProfile profiling controls and rotating log output for NeuralImage.

Environment switches:
  NEURALIMAGE_PROFILE=0 disables all neuralimage profiling.
  NEURALIMAGE_PROFILE=1 enables profiling unless a narrower switch overrides it.
  NEURALIMAGE_PROFILE_<KIND>=0/1 controls a specific profiler.

The bundled ``neuralimage.ini`` is the primary source of defaults. Environment
variables remain supported as deployment/debug overrides.
"""

from __future__ import annotations

import cProfile
import io
import logging
import os
import pstats
import re
from contextlib import contextmanager
from datetime import datetime
from logging import FileHandler
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Iterator

from .runtime_config import (
    config_bool,
    config_int,
    config_string,
    load_runtime_config,
)

PROFILE_ENV_TRUE = {"1", "true", "yes", "on"}
PROFILE_ENV_FALSE = {"0", "false", "no", "off"}

MAX_PROFILE_LOG_FILES = 500
DEFAULT_TOP_LINES = 40

_PROFILE_LOGGER_NAME = "neuralimage.profiling.output"
_PROFILE_LOG_LOCK = Lock()
_PROFILE_LOG_HANDLER: FileHandler | None = None
_PROFILE_BASE_DIR: Path | None = None
_LOGGER = logging.getLogger(__name__)


def _new_profile_log_path(configured_path: Path) -> Path:
    # Build on the full file name (not Path.stem): the default name
    # ``profiling.log`` has a dot in its stem, and ``.stem`` would drop it.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return configured_path.with_name(
        f"{configured_path.name}-{timestamp}-p{os.getpid()}{configured_path.suffix}"
    )


def _profile_log_files(configured_path: Path) -> list[Path]:
    generated = list(configured_path.parent.glob(f"{configured_path.name}-*-p*{configured_path.suffix}"))
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
            _LOGGER.warning("Cannot remove old NeuralImage profiling log: %s", stale_path, exc_info=True)


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
    for name in ("NEURALIMAGE_PROFILE", "NEURALIMAGE_PROFILING", "NEURALIMAGE_PROFILE_ALL"):
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
    env_name = f"NEURALIMAGE_PROFILE_{kind.upper()}"
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
    specific = _env_int(f"NEURALIMAGE_PROFILE_{kind.upper()}_TOP", 0, minimum=0)
    if specific > 0:
        return specific
    common_env = _env_int("NEURALIMAGE_PROFILE_TOP", 0, minimum=0)
    if common_env > 0:
        return common_env
    return config_int(
        "profiling",
        f"{kind}_top_lines",
        config_int("profiling", "top_lines", default, minimum=1),
        minimum=1,
    )


def set_profile_base_dir(path: Path | None) -> None:
    """Set the process's base directory for the profiling log (or clear it)."""

    global _PROFILE_BASE_DIR
    _PROFILE_BASE_DIR = path


def profiling_log_path() -> Path:
    """Resolve the configured profiling log path against the base dir or cwd."""

    log_file = config_string("profiling", "log_file", "profiling.log")
    path = Path(log_file)
    if path.is_absolute():
        return path
    base_dir = _PROFILE_BASE_DIR if _PROFILE_BASE_DIR is not None else Path.cwd()
    return base_dir / path


def _format_cprofile_report(
    kind: str,
    *,
    label: str,
    extra: str,
    wall_ms: float,
    profiler: cProfile.Profile | None,
) -> list[str]:
    summary = f"[neuralimage {label}] {extra} wall_ms={wall_ms:.3f}".rstrip()
    if profiler is None:
        summary += " cprofile_skipped=yes"
    lines = [summary]
    if profiler is not None:
        top = profiling_top_lines(kind, DEFAULT_TOP_LINES)
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(top)
        lines.append(f"[neuralimage {label} stats] top={top}\n{stream.getvalue()}".rstrip())
    return lines


@contextmanager
def cprofile_region(kind: str, *, label: str, extra: str = "") -> Iterator[None]:
    if not profiling_enabled(kind, default=False):
        yield
        return
    profiler = cProfile.Profile()
    active = try_enable_profiler(profiler)
    started = perf_counter()
    try:
        yield
    finally:
        wall_ms = (perf_counter() - started) * 1000.0
        if active:
            try_disable_profiler(profiler)
        write_profile_report(*_format_cprofile_report(kind, label=label, extra=extra, wall_ms=wall_ms, profiler=profiler if active else None))


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
                _LOGGER.exception("Cannot open NeuralImage profiling log: %s", path)
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


def training_profiling_enabled() -> bool:
    return profiling_enabled("training")


def recognition_profiling_enabled() -> bool:
    return profiling_enabled("recognition")


def training_top_lines() -> int:
    return profiling_top_lines("training", DEFAULT_TOP_LINES)


def recognition_top_lines() -> int:
    return profiling_top_lines("recognition", DEFAULT_TOP_LINES)


def training_profile_stride() -> int:
    env_value = _env_int("NEURALIMAGE_PROFILE_TRAINING_STEP_STRIDE", 0, minimum=0)
    if env_value > 0:
        return env_value
    return config_int("profiling", "training_step_stride", 1, minimum=1)
