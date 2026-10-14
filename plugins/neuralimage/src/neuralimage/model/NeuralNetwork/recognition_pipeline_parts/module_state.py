"""recognition_pipeline: module state."""
from __future__ import annotations
from typing import Any, Callable

Publisher = Callable[[str, Any], None]

MemoryMetricsCollector = Callable[[], dict[str, float] | None]

_QUEUE_EMPTY = object()
