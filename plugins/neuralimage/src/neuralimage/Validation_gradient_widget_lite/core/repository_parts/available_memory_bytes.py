"""repository: available memory bytes."""
from __future__ import annotations
import os
import math
import ctypes
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    FrameIdentity,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import (
    ANALYTICS_BATCH_TARGETS_PER_WORKER,
    ANALYTICS_MAX_BATCH_SIZE,
    ANALYTICS_MEMORY_FRACTION,
    ANALYTICS_WORKER_ENV,
    EPS,
)

class BuildCancelledError(RuntimeError):
    """Signal cooperative cancellation during matrix build or analytics."""

@dataclass(frozen=True, slots=True)
class _PredictionPoint:
    """Compact point feature used by point geometry mode."""

    x: float
    y: float
    score: float
    peak_intensity: float
    local_contrast: float
    blob_score: float
    local_snr: float
    radius: float
    spot_area: float

@dataclass(frozen=True, slots=True)
class _PredictionRegionSummary:
    """Compact region summary used for geometry auto-selection."""

    area_fraction: float
    mean_area: float

@dataclass(frozen=True, slots=True)
class _PredictionView:
    """Self-contained prediction view for the extended widget."""

    model_name: str
    pred_gray: np.ndarray
    pred_bin: np.ndarray
    points: tuple[_PredictionPoint, ...]
    region_summary: _PredictionRegionSummary

@dataclass(frozen=True, slots=True)
class _OriginalFrameFeatures:
    """Scalar features extracted from the original grayscale frame."""

    mean_brightness: float
    contrast: float
    entropy: float
    blur_score: float
    noise_score: float
    edge_density: float
    local_peak_density: float
    dynamic_range: float
    saturation_ratio: float

def extract_frame_id(value: str) -> int:
    stem = Path(str(value)).stem
    token = stem.rsplit("_", 1)[-1]
    return int(token) if token.isdigit() else 0

def build_frame_identity(key: str, fallback_frame_id: int) -> FrameIdentity:
    path = Path(key)
    sequence_id = path.parent.as_posix() if path.parent.as_posix() not in {"", "."} else None
    try:
        frame_id = extract_frame_id(path.name)
    except Exception:
        frame_id = int(fallback_frame_id)
    return FrameIdentity(
        frame_id=frame_id,
        base_id=frame_id,
        tile_x=None,
        tile_y=None,
        source_key=key,
        sequence_id=sequence_id,
    )

def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, float(value))))

def _normalize_ratio(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(value / (1.0 + value))

def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    valid = [(float(v), float(w)) for v, w in pairs if np.isfinite(v) and np.isfinite(w) and w > 0.0]
    if not valid:
        return 0.0
    numerator = sum(v * w for v, w in valid)
    denominator = sum(w for _v, w in valid)
    return float(numerator / max(EPS, denominator))

def _analytics_batch_size(total_items: int, worker_count: int) -> int:
    total = max(1, int(total_items))
    workers = max(1, int(worker_count))
    target_batches = max(1, workers * ANALYTICS_BATCH_TARGETS_PER_WORKER)
    return max(1, min(ANALYTICS_MAX_BATCH_SIZE, math.ceil(total / target_batches)))

def _use_thread_pool_for_analytics() -> bool:
    # ProcessPoolExecutor is fragile inside a PyQt worker thread on Windows:
    # child process spawn/import or pickle stalls look like a frozen progress bar.
    return os.name == "nt"

def _analytics_worker_env_override() -> int | None:
    value = os.environ.get(ANALYTICS_WORKER_ENV)
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    return parsed if parsed > 0 else None

def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages > 0:
            return page_size * available_pages
    except Exception:
        return None
    return None

def _estimated_analytics_worker_memory_bytes(
    *,
    analysis_max_side: int | None,
    model_count: int,
    include_model_confidence: bool,
    include_pairwise_metrics: bool,
    include_model_metrics: bool,
) -> int:
    side = int(analysis_max_side or 2048)
    side = max(64, min(side, 8192))
    pixels = int(side) * int(side)
    models = max(1, int(model_count))
    layer_factor = 4 + min(models, 8)
    if include_pairwise_metrics:
        layer_factor += min(models * 2, 12)
    if include_model_metrics:
        layer_factor += min(models, 8)
    if include_model_confidence:
        layer_factor += min(models * 3, 16)
    return max(64 * 1024 * 1024, int(pixels * layer_factor * 4))

def _memory_limited_worker_count(
    *,
    analysis_max_side: int | None,
    model_count: int,
    include_model_confidence: bool,
    include_pairwise_metrics: bool,
    include_model_metrics: bool,
) -> int | None:
    available = _available_memory_bytes()
    if available is None or available <= 0:
        return None
    per_worker = _estimated_analytics_worker_memory_bytes(
        analysis_max_side=analysis_max_side,
        model_count=model_count,
        include_model_confidence=include_model_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
        include_model_metrics=include_model_metrics,
    )
    budget = max(per_worker, int(float(available) * ANALYTICS_MEMORY_FRACTION))
    return max(1, int(budget // max(1, per_worker)))
