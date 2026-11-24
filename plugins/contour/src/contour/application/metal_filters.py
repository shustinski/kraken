"""Which recognized-metal filters are switched on."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

FILTER_MIN_OBJECT_SOURCE_CONTRAST = "min_object_source_contrast"
FILTER_MIN_OBJECT_RIM_CONTRAST = "min_object_rim_contrast"
FILTER_MIN_OBJECT_RIM_AREA_FRACTION = "min_object_rim_area_fraction"
FILTER_MIN_AREA = "min_area"
FILTER_MAX_AREA = "max_area"
FILTER_MIN_PERIMETER = "min_perimeter"
FILTER_MAX_PERIMETER = "max_perimeter"
FILTER_MIN_TRACE_LENGTH = "min_trace_length"
FILTER_MIN_TRACE_WIDTH = "min_trace_width"
FILTER_MAX_TRACE_WIDTH = "max_trace_width"
FILTER_MIN_POINTS = "min_points"
FILTER_MIN_POLYGON_ANGLE = "min_polygon_angle"

METAL_FILTER_KEYS: tuple[str, ...] = (
    FILTER_MIN_OBJECT_SOURCE_CONTRAST,
    FILTER_MIN_OBJECT_RIM_CONTRAST,
    FILTER_MIN_OBJECT_RIM_AREA_FRACTION,
    FILTER_MIN_AREA,
    FILTER_MAX_AREA,
    FILTER_MIN_PERIMETER,
    FILTER_MAX_PERIMETER,
    FILTER_MIN_TRACE_LENGTH,
    FILTER_MIN_TRACE_WIDTH,
    FILTER_MAX_TRACE_WIDTH,
    FILTER_MIN_POINTS,
    FILTER_MIN_POLYGON_ANGLE,
)

# Out of the box only area and point count reject recognized objects.
DEFAULT_ENABLED_METAL_FILTERS: tuple[str, ...] = (FILTER_MIN_AREA, FILTER_MAX_AREA, FILTER_MIN_POINTS)


def normalize_enabled_metal_filters(value: Any) -> tuple[str, ...]:
    """Known filter keys in canonical order; ``None`` (never saved) means the defaults."""

    if value is None:
        return DEFAULT_ENABLED_METAL_FILTERS
    if isinstance(value, str):
        value = [part for part in value.split(",") if part.strip()]
    if not isinstance(value, Iterable):
        return DEFAULT_ENABLED_METAL_FILTERS
    requested = {str(item).strip() for item in value}
    return tuple(key for key in METAL_FILTER_KEYS if key in requested)
