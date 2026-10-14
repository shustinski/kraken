"""_core: module state."""
from __future__ import annotations
from typing import Any

_VIA_DETECTION_CACHE: dict[tuple[str, str, str], Any] = {}

_VIA_DETECTION_CACHE_MAX_ITEMS = 8
