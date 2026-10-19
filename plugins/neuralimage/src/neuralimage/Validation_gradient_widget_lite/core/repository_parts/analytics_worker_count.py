"""repository: analytics worker count."""
from __future__ import annotations
from typing import TYPE_CHECKING
import os
from functools import lru_cache
from pathlib import Path
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    NATURAL_SPLIT_PATTERN,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    GeometryMode,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import (
    WINDOWS_PROCESS_ANALYTICS_MAX_WORKERS,
    WINDOWS_THREAD_ANALYTICS_MAX_WORKERS,
    WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS,
    _LAST_ANALYTICS_WORKER_PLAN,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import (
    BuildCancelledError,
    _analytics_worker_env_override,
    _memory_limited_worker_count,
)

def _cpu_limited_thread_worker_count(
    *,
    cpu_limit: int,
    analysis_max_side: int | None,
    include_model_confidence: bool,
) -> int:
    side_limit = int(analysis_max_side or 0)
    if include_model_confidence:
        if side_limit == 0 or side_limit > 1024:
            return min(cpu_limit, WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS)
        return min(cpu_limit, max(WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS, cpu_limit // 2))
    if side_limit > 0 and side_limit <= 512:
        return min(cpu_limit, max(16, cpu_limit))
    if side_limit > 0 and side_limit <= 1024:
        return min(cpu_limit, WINDOWS_THREAD_ANALYTICS_MAX_WORKERS)
    return min(cpu_limit, max(16, WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS))

def _analytics_worker_count(
    max_workers: int,
    *,
    geometry_mode: GeometryMode,
    analysis_max_side: int | None,
    include_model_confidence: bool,
    include_pairwise_metrics: bool = True,
    include_model_metrics: bool = True,
    model_count: int = 1,
    use_thread_pool: bool,
) -> int:
    requested = max(1, int(max_workers))
    override = _analytics_worker_env_override()
    if override is not None:
        return min(requested, max(1, int(override)))
    if os.name != "nt" or geometry_mode == GeometryMode.POINT:
        return requested
    cpu_limit = max(1, os.cpu_count() or requested)
    if use_thread_pool:
        safe_limit = _cpu_limited_thread_worker_count(
            cpu_limit=cpu_limit,
            analysis_max_side=analysis_max_side,
            include_model_confidence=include_model_confidence,
        )
        memory_limit = _memory_limited_worker_count(
            analysis_max_side=analysis_max_side,
            model_count=model_count,
            include_model_confidence=include_model_confidence,
            include_pairwise_metrics=include_pairwise_metrics,
            include_model_metrics=include_model_metrics,
        )
        if memory_limit is not None:
            safe_limit = min(safe_limit, memory_limit)
        return min(requested, max(1, safe_limit))
    safe_limit = min(cpu_limit, WINDOWS_PROCESS_ANALYTICS_MAX_WORKERS)
    side_limit = int(analysis_max_side or 0)
    if include_model_confidence and side_limit > 0 and side_limit <= 768:
        safe_limit = min(cpu_limit, 5)
    return min(requested, max(1, safe_limit))

def _remember_analytics_worker_plan(
    *,
    requested: int,
    selected: int,
    use_thread_pool: bool,
    geometry_mode: GeometryMode,
    analysis_max_side: int | None,
    model_count: int,
    include_model_confidence: bool,
    include_pairwise_metrics: bool,
    include_model_metrics: bool,
) -> None:
    _LAST_ANALYTICS_WORKER_PLAN.clear()
    _LAST_ANALYTICS_WORKER_PLAN.update(
        {
            "requested": int(requested),
            "selected": int(selected),
            "use_thread_pool": bool(use_thread_pool),
            "geometry_mode": str(getattr(geometry_mode, "value", geometry_mode)),
            "analysis_max_side": None if analysis_max_side is None else int(analysis_max_side),
            "model_count": int(model_count),
            "include_model_confidence": bool(include_model_confidence),
            "include_pairwise_metrics": bool(include_pairwise_metrics),
            "include_model_metrics": bool(include_model_metrics),
        }
    )

def last_analytics_worker_plan() -> dict[str, object]:
    return dict(_LAST_ANALYTICS_WORKER_PLAN)

def _unique_path_lookup(index: dict[str, Path], key_fn) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in index.values():
        key = str(key_fn(path)).lower()
        if key in duplicates:
            continue
        if key in lookup:
            lookup.pop(key, None)
            duplicates.add(key)
            continue
        lookup[key] = path
    return lookup

def _build_folder_path_lookup(index: dict[str, Path]) -> _FolderPathLookup:
    return (
        index,
        _unique_path_lookup(index, lambda path: Path(path).name),
        _unique_path_lookup(index, lambda path: Path(path).stem),
    )

def _resolve_aux_path_from_lookup(key: str, lookup: _FolderPathLookup) -> Path | None:
    index, name_lookup, stem_lookup = lookup
    exact = index.get(key)
    if exact is not None:
        return exact
    name = Path(key).name.lower()
    stem = Path(key).stem.lower()
    name_match = name_lookup.get(name)
    if name_match is not None:
        return name_match
    stem_match = stem_lookup.get(stem)
    if stem_match is not None:
        return stem_match
    return None

def _resolve_aux_path(key: str, index: dict[str, Path]) -> Path | None:
    return _resolve_aux_path_from_lookup(key, _build_folder_path_lookup(index))

@lru_cache(maxsize=65536)
def natural_sort_key(value: str) -> tuple[object, ...]:
    """Split a string into digit and text chunks for natural sorting."""

    parts = NATURAL_SPLIT_PATTERN.split(str(value).lower())
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)

def iter_image_paths(folder: Path, *, recursive: bool, extensions: tuple[str, ...]) -> list[Path]:
    """Return image paths from one folder using natural sorting."""

    normalized_extensions = {str(ext).lower() for ext in extensions}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    return sorted(paths, key=lambda item: natural_sort_key(item.as_posix()))

def build_folder_index(
    folder: Path,
    *,
    recursive: bool,
    extensions: tuple[str, ...],
    cancel_check=None,
    progress_callback=None,
    progress_interval: int = 2048,
) -> dict[str, Path]:
    """Index frame files in one folder by relative path."""

    folder = Path(folder)
    normalized_extensions = {str(ext).lower() for ext in extensions}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths: list[Path] = []
    accepted = 0
    interval = max(1, int(progress_interval))
    for path in iterator:
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        if not path.is_file() or path.suffix.lower() not in normalized_extensions:
            continue
        paths.append(path)
        accepted += 1
        if progress_callback is not None and accepted % interval == 0:
            progress_callback(accepted, 0, path.name)
    paths.sort(key=lambda item: natural_sort_key(item.as_posix()))
    index: dict[str, Path] = {}
    for image_path in paths:
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        index[image_path.relative_to(folder).as_posix()] = image_path
    if progress_callback is not None:
        progress_callback(len(index), len(index), Path(folder).name)
    return index

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import _FolderPathLookup
