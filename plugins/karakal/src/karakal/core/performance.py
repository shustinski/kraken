"""Central performance settings shared by Karakal workers and UI."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .backend_constants import CACHE_DIR


class ProfilingMode(StrEnum):
    OFF = "off"
    SUMMARY = "summary"
    DETAILED = "detailed"
    TRACE = "trace"


def _logical_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _default_cpu_workers() -> int:
    return max(1, min(4, (_logical_cpu_count() + 1) // 2))


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    """Validated runtime settings; expensive diagnostics are disabled by default."""

    cpu_workers: int = 0
    io_workers: int = 4
    batch_size: int = 8
    parallel_enabled: bool = True
    sequential_debug_mode: bool = False
    opencv_threads: int = 1
    ram_cache_limit_mb: int = 512
    disk_cache_limit_mb: int = 4096
    preview_cache_limit_mb: int = 256
    tile_cache_limit_mb: int = 256
    progress_update_interval_ms: int = 100
    profiling_mode: ProfilingMode = ProfilingMode.OFF
    profiling_output_directory: str = ""
    profiling_export_json: bool = True
    profiling_export_csv: bool = True
    profiling_export_markdown: bool = True
    profiling_trace_enabled: bool = False
    profiling_trace_frame_limit: int = 100
    profiling_slow_frame_threshold_ms: float = 250.0
    profiling_ui_refresh_interval_ms: int = 300
    profiling_collect_memory: bool = False
    profiling_collect_cpu: bool = False
    profiling_collect_worker_stats: bool = True
    profiling_collect_cache_stats: bool = True
    profiling_keep_last_runs: int = 10

    def __post_init__(self) -> None:
        cpus = _logical_cpu_count()
        object.__setattr__(self, "cpu_workers", max(1, min(cpus, int(self.cpu_workers or _default_cpu_workers()))))
        object.__setattr__(self, "io_workers", max(1, min(64, int(self.io_workers))))
        object.__setattr__(self, "batch_size", max(1, min(4096, int(self.batch_size))))
        object.__setattr__(self, "opencv_threads", max(1, min(cpus, int(self.opencv_threads))))
        for name in ("ram_cache_limit_mb", "disk_cache_limit_mb", "preview_cache_limit_mb", "tile_cache_limit_mb"):
            object.__setattr__(self, name, max(1, int(getattr(self, name))))
        object.__setattr__(self, "progress_update_interval_ms", max(25, int(self.progress_update_interval_ms)))
        object.__setattr__(self, "profiling_trace_frame_limit", max(1, int(self.profiling_trace_frame_limit)))
        object.__setattr__(
            self, "profiling_slow_frame_threshold_ms", max(0.0, float(self.profiling_slow_frame_threshold_ms))
        )
        object.__setattr__(
            self, "profiling_ui_refresh_interval_ms", max(200, int(self.profiling_ui_refresh_interval_ms))
        )
        object.__setattr__(self, "profiling_keep_last_runs", max(1, int(self.profiling_keep_last_runs)))
        mode = self.profiling_mode
        if not isinstance(mode, ProfilingMode):
            try:
                mode = ProfilingMode(str(mode).strip().lower())
            except ValueError:
                mode = ProfilingMode.OFF
            object.__setattr__(self, "profiling_mode", mode)

    @property
    def profiling_directory(self) -> Path:
        value = str(self.profiling_output_directory or "").strip()
        return Path(value).expanduser() if value else CACHE_DIR / "profiles"

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profiling_mode"] = self.profiling_mode.value
        return payload


_ENV_NAMES: dict[str, str] = {
    "cpu_workers": "KARAKAL_CPU_WORKERS",
    "io_workers": "KARAKAL_IO_WORKERS",
    "batch_size": "KARAKAL_BATCH_SIZE",
    "parallel_enabled": "KARAKAL_PARALLEL_ENABLED",
    "sequential_debug_mode": "KARAKAL_SEQUENTIAL_DEBUG_MODE",
    "opencv_threads": "KARAKAL_OPENCV_THREADS",
    "ram_cache_limit_mb": "KARAKAL_RAM_CACHE_LIMIT_MB",
    "disk_cache_limit_mb": "KARAKAL_DISK_CACHE_LIMIT_MB",
    "preview_cache_limit_mb": "KARAKAL_PREVIEW_CACHE_LIMIT_MB",
    "tile_cache_limit_mb": "KARAKAL_TILE_CACHE_LIMIT_MB",
    "progress_update_interval_ms": "KARAKAL_PROGRESS_UPDATE_INTERVAL_MS",
    "profiling_mode": "KARAKAL_PROFILING_MODE",
    "profiling_output_directory": "KARAKAL_PROFILING_OUTPUT_DIRECTORY",
    "profiling_export_json": "KARAKAL_PROFILING_EXPORT_JSON",
    "profiling_export_csv": "KARAKAL_PROFILING_EXPORT_CSV",
    "profiling_export_markdown": "KARAKAL_PROFILING_EXPORT_MARKDOWN",
    "profiling_trace_enabled": "KARAKAL_PROFILING_TRACE_ENABLED",
    "profiling_trace_frame_limit": "KARAKAL_PROFILING_TRACE_FRAME_LIMIT",
    "profiling_slow_frame_threshold_ms": "KARAKAL_PROFILING_SLOW_FRAME_THRESHOLD_MS",
    "profiling_ui_refresh_interval_ms": "KARAKAL_PROFILING_UI_REFRESH_INTERVAL_MS",
    "profiling_collect_memory": "KARAKAL_PROFILING_COLLECT_MEMORY",
    "profiling_collect_cpu": "KARAKAL_PROFILING_COLLECT_CPU",
    "profiling_collect_worker_stats": "KARAKAL_PROFILING_COLLECT_WORKER_STATS",
    "profiling_collect_cache_stats": "KARAKAL_PROFILING_COLLECT_CACHE_STATS",
    "profiling_keep_last_runs": "KARAKAL_PROFILING_KEEP_LAST_RUNS",
}

_LEGACY_ENV_NAMES: dict[str, str] = {
    "cpu_workers": "KARAKAL_GRID_INSPECTION_WORKERS",
    "batch_size": "KARAKAL_GRID_INSPECTION_CHUNK_SIZE",
    "opencv_threads": "KARAKAL_GRID_INSPECTION_OPENCV_THREADS",
}


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce(name: str, value: object, default: object) -> object:
    try:
        if name == "profiling_mode":
            return ProfilingMode(str(value).strip().lower())
        if isinstance(default, bool):
            return _as_bool(value, default)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        return str(value)
    except (TypeError, ValueError):
        return default


def load_performance_config(
    settings: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> PerformanceConfig:
    """Load settings with environment > persisted settings > defaults precedence."""

    defaults = PerformanceConfig()
    source = dict(settings or {})
    env = os.environ if environ is None else environ
    values: dict[str, object] = {}
    for name, default in defaults.to_payload().items():
        value = source.get(name, default)
        legacy_name = _LEGACY_ENV_NAMES.get(name)
        if legacy_name and legacy_name in env:
            value = env[legacy_name]
        env_name = _ENV_NAMES.get(name)
        if env_name and env_name in env:
            value = env[env_name]
        values[name] = _coerce(name, value, default)
    legacy_execution = str(env.get("KARAKAL_GRID_INSPECTION_EXECUTION", "")).strip().lower()
    if legacy_execution == "sequential":
        values["sequential_debug_mode"] = True
    return PerformanceConfig(**values)


__all__ = ["PerformanceConfig", "ProfilingMode", "load_performance_config"]
