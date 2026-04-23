"""Via-candidate detection: debug-map builders and edge-method resolvers.

All symbols here are currently implemented in :mod:`._core`; this module provides
a stable import surface for callers that want to reach the via-detection layer
specifically.
"""

from __future__ import annotations

from ._core import (
    _resolve_conductor_edge_method,
    _resolve_via_edge_method,
    _via_grayscale,
    build_detection_debug_maps,
)

__all__ = [
    "_resolve_conductor_edge_method",
    "_resolve_via_edge_method",
    "_via_grayscale",
    "build_detection_debug_maps",
]
