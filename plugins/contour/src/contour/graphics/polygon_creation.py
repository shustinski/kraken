"""Validation for committing user-drawn closed polygons (pure, unit-tested)."""

from __future__ import annotations

from ..domain import compute_polygon_metrics
from .geometry import is_valid_closed_polygon_ring

# Degenerate / noise-level shapes in scene pixel units (consistent with editor float coords).
POLYGON_COMMIT_MIN_ABS_AREA_PX2 = 1e-6
POLYGON_COMMIT_MIN_VERTICES = 3

# Reason strings used by PolygonEditorScene → i18n logs.
POLYGON_COMMIT_TOO_FEW_VERTICES = "too_few_vertices"
POLYGON_COMMIT_INVALID_RING = "invalid_ring"
POLYGON_COMMIT_TOO_SMALL_AREA = "too_small_area"


def polygon_commit_acceptability(
    points: list[tuple[float, float]],
    *,
    min_vertices: int = POLYGON_COMMIT_MIN_VERTICES,
    min_abs_area_px2: float = POLYGON_COMMIT_MIN_ABS_AREA_PX2,
) -> tuple[bool, str | None]:
    """Return whether a closed ring is safe to store; second value is a stable reason key."""
    if len(points) < min_vertices:
        return False, POLYGON_COMMIT_TOO_FEW_VERTICES
    if not is_valid_closed_polygon_ring(points):
        return False, POLYGON_COMMIT_INVALID_RING
    area, _perimeter, _bbox = compute_polygon_metrics(points)
    if abs(float(area)) < float(min_abs_area_px2):
        return False, POLYGON_COMMIT_TOO_SMALL_AREA
    return True, None
