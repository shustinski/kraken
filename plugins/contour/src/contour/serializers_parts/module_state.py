"""serializers: module state."""
from __future__ import annotations
import threading
from contour.domain import PolygonData

_CIF_PARSE_CACHE: dict[tuple[str, int, int], tuple[str | None, tuple[int, int] | None, list[PolygonData]]] = {}

_CIF_PARSE_CACHE_LOCK = threading.Lock()

_CIF_PARSE_CACHE_MAX_ENTRIES = 64

CIF_CUTOUT_DISPLAY_MARKER = "( CONTOUR cutout_display );"

_BOUNDARY_SNAP_NEIGHBOR_WINDOW = 8

_HoleLinkDetour = tuple[
    float,
    tuple[float, float],
    list[tuple[float, float]],
    int,
    int,
]
