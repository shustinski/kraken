"""repository: module state."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path

EPS = 1e-8

ANALYTICS_MAX_BATCH_SIZE = 16

ANALYTICS_BATCH_TARGETS_PER_WORKER = 32

ANALYTICS_WAIT_TIMEOUT_SECONDS = 0.25

ANALYTICS_STALL_TIMEOUT_SECONDS = 180.0

ANALYTICS_STALL_PROGRESS_SECONDS = 5.0

WINDOWS_THREAD_ANALYTICS_MAX_WORKERS = 64

WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS = 32

WINDOWS_PROCESS_ANALYTICS_MAX_WORKERS = 4

ANALYTICS_WORKER_ENV = "VALIDATION_MATRIX_ANALYTICS_WORKERS"

ANALYTICS_MEMORY_FRACTION = 0.45

ANALYSIS_CACHE_MAX_FILES = 100000

DETAIL_CACHE_MAX_FILES = 20000

CACHE_TRIM_INTERVAL_SECONDS = 300.0

_CACHE_TRIM_LAST_BY_DIR: dict[str, float] = {}

_LAST_ANALYTICS_WORKER_PLAN: dict[str, object] = {}

_FolderPathLookup = tuple[dict[str, Path], dict[str, Path], dict[str, Path]]

DETAIL_PAYLOAD_CACHE_SIZE = 32

_DETAIL_PAYLOAD_MEMORY_CACHE: OrderedDict[str, object] = OrderedDict()

_NEIGHBOR_SLICE_ORDER = (
    (slice(0, -2), slice(0, -2)),
    (slice(0, -2), slice(1, -1)),
    (slice(0, -2), slice(2, None)),
    (slice(1, -1), slice(0, -2)),
    (slice(1, -1), slice(2, None)),
    (slice(2, None), slice(0, -2)),
    (slice(2, None), slice(1, -1)),
    (slice(2, None), slice(2, None)),
)
