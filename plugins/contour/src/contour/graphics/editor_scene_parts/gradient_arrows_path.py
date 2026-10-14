"""editor_scene: gradient arrows path."""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot
from PyQt6.QtGui import (
    QPainterPath,
)
from contour.domain import PolygonData

from contour.graphics.editor_scene_parts.module_state import _ZOOM_QUANTIZED_CHANNELS

@dataclass(slots=True)
class _DeferredGeometryRepairRequest:
    removed_ids: set[int]
    removed_polygons: list[PolygonData]

def _gradient_arrows_path(arrows: list[tuple[float, float, float, float]]) -> QPainterPath:
    path = QPainterPath()
    for origin_x, origin_y, delta_x, delta_y in arrows:
        length = hypot(delta_x, delta_y)
        if length <= 1e-9:
            continue
        ux, uy = delta_x / length, delta_y / length
        nx, ny = -uy, ux
        tip_length = 0.32 * length
        tip_half = 0.18 * length
        end_x = origin_x + delta_x
        end_y = origin_y + delta_y
        shaft_x = end_x - ux * tip_length
        shaft_y = end_y - uy * tip_length
        path.moveTo(origin_x, origin_y)
        path.lineTo(shaft_x, shaft_y)
        path.moveTo(end_x, end_y)
        path.lineTo(end_x - ux * tip_length + nx * tip_half, end_y - uy * tip_length + ny * tip_half)
        path.lineTo(end_x - ux * tip_length - nx * tip_half, end_y - uy * tip_length - ny * tip_half)
        path.closeSubpath()
    return path

def _zoom_quantized_rgba(rgba: int) -> int:
    value = int(rgba)
    return (
        (value & 0xFF000000)
        | (_ZOOM_QUANTIZED_CHANNELS[(value >> 16) & 0xFF] << 16)
        | (_ZOOM_QUANTIZED_CHANNELS[(value >> 8) & 0xFF] << 8)
        | _ZOOM_QUANTIZED_CHANNELS[value & 0xFF]
    )
