"""repository: record payload cache key."""
from __future__ import annotations
import hashlib
import json
import pickle
from time import perf_counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    ANALYSIS_CACHE_DIR,
    ANALYSIS_CACHE_VERSION,
    IMAGE_CACHE_SIZE,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    FrameRecord,
    GeometryMode,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import (
    CACHE_TRIM_INTERVAL_SECONDS,
    _CACHE_TRIM_LAST_BY_DIR,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.analytics_worker_count import (
    _build_folder_path_lookup,
    _resolve_aux_path_from_lookup,
    build_folder_index,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import BuildCancelledError

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mean_filter_3x3 import (
    load_grayscale_image,
    resize_grayscale_image,
)

def _candidate_paths_for_known_key(folder: Path, key: str, extensions: tuple[str, ...]) -> list[Path]:
    relative = Path(key)
    candidates: list[Path] = []
    direct = folder / relative
    candidates.append(direct)
    stem_name = relative.stem
    parent = folder / relative.parent
    seen: set[str] = {str(direct).lower()}
    for extension in extensions:
        normalized = str(extension or '').strip()
        if not normalized:
            continue
        if not normalized.startswith('.'):
            normalized = f'.{normalized}'
        candidate = parent / f'{stem_name}{normalized}'
        marker = str(candidate).lower()
        if marker in seen:
            continue
        seen.add(marker)
        candidates.append(candidate)
    return candidates

def _resolve_model_path_from_known_key(folder: Path, key: str, extensions: tuple[str, ...]) -> Path | None:
    for candidate in _candidate_paths_for_known_key(folder, key, extensions):
        if candidate.is_file():
            return candidate
    return None

def _resolve_model_path_for_key(
    folder: Path,
    key: str,
    extensions: tuple[str, ...],
    fallback_index_cache: dict[str, _FolderPathLookup],
    *,
    recursive: bool,
    cancel_check=None,
    progress_callback=None,
) -> Path | None:
    cache_key = str(folder.resolve())
    lookup = fallback_index_cache.get(cache_key)
    if lookup is not None:
        return _resolve_aux_path_from_lookup(key, lookup)
    resolved = _resolve_model_path_from_known_key(folder, key, extensions)
    if resolved is not None:
        return resolved
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError('Build cancelled')
    index = build_folder_index(
        folder,
        recursive=recursive,
        extensions=extensions,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    lookup = _build_folder_path_lookup(index)
    fallback_index_cache[cache_key] = lookup
    return _resolve_aux_path_from_lookup(key, lookup)

def _fit_shape_to_max_side(shape: tuple[int, int], max_side: int | None) -> tuple[int, int]:
    height, width = (int(shape[0]), int(shape[1]))
    limit = int(max_side or 0)
    if limit <= 0 or max(height, width) <= limit:
        return height, width
    scale = float(limit) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))

def _image_signature(path: Path) -> tuple[str, int, int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)

@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _load_grayscale_image_cached(path_text: str, _mtime_ns: int, _size: int) -> np.ndarray:
    return load_grayscale_image(Path(path_text))

@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _load_resized_grayscale_image_cached(path_text: str, _mtime_ns: int, _size: int, target_shape: tuple[int, int]) -> np.ndarray:
    source = load_grayscale_image(Path(path_text))
    return resize_grayscale_image(source, target_shape)

def _clear_runtime_image_caches() -> None:
    """Release in-memory grayscale caches used during batch analytics."""

    _load_grayscale_image_cached.cache_clear()
    _load_resized_grayscale_image_cached.cache_clear()

def _load_optional_gray(path_text: str | None, target_shape: tuple[int, int] | None = None, max_side: int | None = None) -> np.ndarray | None:
    if not path_text:
        return None
    source_path = Path(path_text)
    source = _load_grayscale_image_cached(*_image_signature(source_path))
    limited_shape = _fit_shape_to_max_side(tuple(int(v) for v in source.shape), max_side)
    final_shape = tuple(int(v) for v in target_shape) if target_shape is not None else limited_shape
    if tuple(int(v) for v in source.shape) == final_shape:
        return np.asarray(source, dtype=np.uint8)
    return np.asarray(_load_resized_grayscale_image_cached(*_image_signature(source_path), final_shape), dtype=np.uint8)

def _path_signature(path_text: str | None) -> tuple[str, int, int] | None:
    if not path_text:
        return None
    return _image_signature(Path(path_text))

def _record_payload_cache_key(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    *,
    include_model_confidence: bool,
    include_pairwise_metrics: bool,
    include_model_metrics: bool,
) -> str:
    payload = {
        "version": ANALYSIS_CACHE_VERSION,
        "record_key": record.key,
        "analysis_max_side": int(analysis_max_side or 0),
        "geometry_mode": geometry_mode.value,
        "point_match_radius": float(point_match_radius),
        "boundary_radius": int(boundary_radius),
        "confidence_uncertainty_delta": float(confidence_uncertainty_delta),
        "point_confidence_radius": int(point_confidence_radius),
        "polygon_confidence_summary": str(polygon_confidence_summary),
        "include_model_confidence": bool(include_model_confidence),
        "include_pairwise_metrics": bool(include_pairwise_metrics),
        "include_model_metrics": bool(include_model_metrics),
        "original": _path_signature(record.original_path),
        "gt": _path_signature(record.gt_path),
        "models": [
            {
                "model_id": spec.model_id,
                "threshold": float(spec.threshold),
                "mask": _path_signature(record.model_mask_paths.get(spec.model_id)),
                "prob": _path_signature(record.model_prob_paths.get(spec.model_id)),
            }
            for spec in model_specs
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def _record_payload_cache_path(cache_key: str) -> Path:
    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_CACHE_DIR / f"{cache_key}.pickle"

def _trim_pickle_cache_dir(cache_dir: Path, *, max_files: int) -> None:
    directory_key = str(Path(cache_dir))
    now = perf_counter()
    last_trim = _CACHE_TRIM_LAST_BY_DIR.get(directory_key, 0.0)
    if now - last_trim < CACHE_TRIM_INTERVAL_SECONDS:
        return
    _CACHE_TRIM_LAST_BY_DIR[directory_key] = now
    try:
        cache_files = list(Path(cache_dir).glob("*.pickle"))
    except Exception:
        return
    extra_count = len(cache_files) - int(max_files)
    if extra_count <= 0:
        return
    try:
        removable = sorted(cache_files, key=lambda path: path.stat().st_mtime_ns)[:extra_count]
    except Exception:
        return
    for path in removable:
        try:
            path.unlink()
        except Exception:
            continue

def _load_cached_record_payload(cache_key: str) -> dict[str, object] | None:
    cache_path = _record_payload_cache_path(cache_key)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
