"""matrix_view: store subpixel grid disk."""
from __future__ import annotations
import hashlib
import pickle
import time
from pathlib import Path
import numpy as np
from ...core.domain import FrameRecord
from ...core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import (
    SUBPIXEL_GRID_DISK_CACHE_DIR,
    SUBPIXEL_GRID_DISK_CACHE_MAX_FILES,
    SUBPIXEL_GRID_DISK_CACHE_TRIM_INTERVAL_SECONDS,
    SUBPIXEL_GRID_DISK_CACHE_VERSION,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.score_fill_color import _record_path_signature

def _subpixel_cache_key_for_record(
    record: FrameRecord,
    spec: SubpixelGridSpec,
    *,
    aggregation: str,
    metric_key: str | None,
    comparison_mode,
) -> tuple[object, ...]:
    return (
        SUBPIXEL_GRID_DISK_CACHE_VERSION,
        str(record.key),
        str(spec.mode),
        int(spec.rows),
        int(spec.columns),
        int(spec.tile_width),
        int(spec.tile_height),
        int(spec.overlap),
        str(aggregation or "mean"),
        str(metric_key or ""),
        str(getattr(comparison_mode, "value", comparison_mode)),
        _record_path_signature(record),
    )

def _subpixel_disk_cache_path(cache_key: tuple[object, ...]) -> Path:
    digest = hashlib.sha1(repr(cache_key).encode("utf-8", errors="ignore")).hexdigest()
    return SUBPIXEL_GRID_DISK_CACHE_DIR / f"{digest}.pickle"

def _load_subpixel_grid_from_disk(cache_key: tuple[object, ...]) -> SubpixelGrid | None:
    cache_path = _subpixel_disk_cache_path(cache_key)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        spec = SubpixelGridSpec(**payload["spec"]).normalized()
        return SubpixelGrid(
            spec=spec,
            values=np.asarray(payload["values"], dtype=np.float32),
            confidences=None if payload.get("confidences") is None else np.asarray(payload["confidences"], dtype=np.float32),
            aggregation=str(payload.get("aggregation") or "mean"),
            value_kind=str(payload.get("value_kind") or "value"),
        )
    except Exception:
        return None

def _trim_subpixel_grid_disk_cache() -> None:
    global _subpixel_grid_disk_cache_last_trim
    now = time.monotonic()
    if now - _subpixel_grid_disk_cache_last_trim < SUBPIXEL_GRID_DISK_CACHE_TRIM_INTERVAL_SECONDS:
        return
    _subpixel_grid_disk_cache_last_trim = now
    try:
        cache_files = list(SUBPIXEL_GRID_DISK_CACHE_DIR.glob("*.pickle"))
    except Exception:
        return
    extra_count = len(cache_files) - int(SUBPIXEL_GRID_DISK_CACHE_MAX_FILES)
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

def _store_subpixel_grid_to_disk(cache_key: tuple[object, ...], grid: SubpixelGrid) -> None:
    try:
        SUBPIXEL_GRID_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _subpixel_disk_cache_path(cache_key)
        tmp_path = cache_path.with_suffix(".tmp")
        payload = {
            "spec": {
                "rows": int(grid.spec.rows),
                "columns": int(grid.spec.columns),
                "mode": str(grid.spec.mode),
                "tile_width": int(grid.spec.tile_width),
                "tile_height": int(grid.spec.tile_height),
                "overlap": int(grid.spec.overlap),
            },
            "values": np.asarray(grid.values, dtype=np.float32),
            "confidences": None if grid.confidences is None else np.asarray(grid.confidences, dtype=np.float32),
            "aggregation": str(grid.aggregation),
            "value_kind": str(grid.value_kind),
        }
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        _trim_subpixel_grid_disk_cache()
    except Exception:
        return
