"""serializers: cif paint mask ring."""
from __future__ import annotations
import os
import numpy as np
from contour.domain import PolygonData, integer_points
from ..domain.polygon_ring import collapse_redundant_polyline_vertices

from contour.serializers_parts.boundary_vertices_by_distance import _dedupe_closed_points

def _remove_out_and_back_spikes(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop zero-area spikes ``A -> B -> A`` that leftover CIF bridges often leave behind."""

    if len(points) < 3:
        return list(points)
    open_points = _dedupe_closed_points(points)
    stack: list[tuple[float, float]] = []
    for point in open_points:
        if len(stack) >= 2 and stack[-2] == point:
            stack.pop()
            continue
        stack.append(point)
    # Closing the ring can also leave a wrap-around spike: last -> first -> second.
    while len(stack) >= 3 and stack[-1] == stack[1]:
        stack.pop(0)
        stack.pop()
    while len(stack) >= 3 and stack[0] == stack[-2]:
        stack.pop()
        stack.pop(0)
    return stack if len(stack) >= 3 else list(points)

def _collapse_recovered_outline(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop leftover keyhole bridges, out-and-back spikes, and collinear stubs."""

    cleaned = _remove_out_and_back_spikes(ring)
    collapsed = collapse_redundant_polyline_vertices(cleaned, closed=True, min_vertices=3)
    return collapsed if len(collapsed) >= 3 else cleaned if len(cleaned) >= 3 else ring

def _cif_paint_mask_from_ring(
    ring: list[tuple[float, float]],
) -> tuple[np.ndarray, int, int]:
    """Rasterize a CIF ``P`` ring with Qt WindingFill (KLayout viewer semantics)."""

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath

    int_points = integer_points(_dedupe_closed_points(ring))
    if len(int_points) < 3:
        return np.zeros((1, 1), dtype=np.uint8), 0, 0
    xs = [point[0] for point in int_points]
    ys = [point[1] for point in int_points]
    left = int(min(xs))
    top = int(min(ys))
    width = max(1, int(max(xs)) - left + 1)
    height = max(1, int(max(ys)) - top + 1)
    path = QPainterPath()
    path.moveTo(float(int_points[0][0] - left), float(int_points[0][1] - top))
    for x_coord, y_coord in int_points[1:]:
        path.lineTo(float(x_coord - left), float(y_coord - top))
    path.closeSubpath()
    path.setFillRule(Qt.FillRule.WindingFill)
    image = QImage(width, height, QImage.Format.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillPath(path, QBrush(QColor(255, 255, 255)))
    finally:
        painter.end()
    mask = np.frombuffer(
        bytes(image.constBits().asarray(image.sizeInBytes())),
        dtype=np.uint8,
    ).reshape((height, image.bytesPerLine()))[:, :width].copy()
    return mask, left, top

def _stamp_cif_paint_ring_on_mask(
    mask: np.ndarray,
    ring: list[tuple[float, float]],
    origin: tuple[int, int],
    *,
    value: int = 255,
) -> None:
    """Paint a CIF ring into ``mask`` using Qt WindingFill (KLayout viewer semantics)."""

    paint_mask, left, top = _cif_paint_mask_from_ring(ring)
    if paint_mask.size == 0 or not np.any(paint_mask):
        return
    x_offset = left - int(origin[0])
    y_offset = top - int(origin[1])
    height, width = paint_mask.shape
    target_x_start = max(0, x_offset)
    target_y_start = max(0, y_offset)
    target_x_stop = min(mask.shape[1], x_offset + width)
    target_y_stop = min(mask.shape[0], y_offset + height)
    if target_x_start >= target_x_stop or target_y_start >= target_y_stop:
        return

    paint_x_start = target_x_start - x_offset
    paint_y_start = target_y_start - y_offset
    paint_x_stop = paint_x_start + (target_x_stop - target_x_start)
    paint_y_stop = paint_y_start + (target_y_stop - target_y_start)
    target = mask[target_y_start:target_y_stop, target_x_start:target_x_stop]
    painted = paint_mask[paint_y_start:paint_y_stop, paint_x_start:paint_x_stop] > 0
    if int(value) == 0:
        target[painted] = 0
        return
    target[painted] = int(value)

def _polygon_uses_authored_cif_paint_ring(polygon: PolygonData) -> bool:
    """True when display must use the authored CIF ring with WindingFill (keyhole split)."""

    if polygon.is_hole or not polygon.cif_paint_ring:
        return False
    return integer_points(polygon.cif_paint_ring) != integer_points(polygon.points)

def _ensure_cif_paint_rings(polygons: list[PolygonData]) -> None:
    """No-op; authored rings are attached during keyhole recovery only."""

    del polygons

def _render_cif_klayout_layer_mask(
    image_size: tuple[int, int],
    polygons: list[PolygonData],
) -> np.ndarray:
    """Rasterize loaded polygons the way KLayout paints independent CIF ``P`` commands."""

    image_width, image_height = image_size
    mask = np.zeros((int(image_height), int(image_width)), dtype=np.uint8)
    for polygon in polygons:
        if polygon.is_hole:
            continue
        ring = polygon.cif_paint_ring if _polygon_uses_authored_cif_paint_ring(polygon) else polygon.points
        _stamp_cif_paint_ring_on_mask(mask, ring, (0, 0), value=255)
    return mask

def _legacy_cut_hole_recovery_enabled() -> bool:
    return str(os.environ.get("CONTOUR_CIF_RECOVER_LEGACY_CUT_HOLES", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

def _nested_outer_punch_on_load_enabled() -> bool:
    """Opt-in only: punch nested outers after CIF load.

    Default is off so filled coverage matches KLayout / CIF semantics (each ``P``
    is painted independently; overlaps OR together; no cross-polygon subtraction).
    Enable via ``CONTOUR_CIF_PUNCH_NESTED_OUTERS`` when an empty window under a
    nested island is required for editing.
    """

    return str(os.environ.get("CONTOUR_CIF_PUNCH_NESTED_OUTERS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

def _ring_coords_from_shapely(coords) -> list[tuple[float, float]]:
    points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in list(coords)[:-1]]
    points = integer_points(points)
    collapsed = collapse_redundant_polyline_vertices(points, closed=True, min_vertices=3)
    return collapsed if len(collapsed) >= 3 else points
