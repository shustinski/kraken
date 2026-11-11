"""How defects are marked for a person: outlines and shapes, not filled boxes.

Every defect type has a shape of its own as well as a colour, so it reads on a small cell and for a
colour-blind eye, and the cell under the mark stays visible:

- broken geometry (and older cell reasons): a solid outline of the object;
- merged cells: the outline and a bar under the cells it joins;
- debris: a ring around the piece (a crumb of a few pixels is seen at once);
- sparse cell: corner brackets outside the cell, together with the outline of its other finding;
- ones/zeros conflict: hatching of the place;
- only one network found: a dotted outline;
- conductor zone: faint hatching, a background layer under the defects.

The picture for programs (core.grid_error_export.render_grid_error_image) keeps its filled boxes,
one picture per layer of errors, several errors of a cell as rings: it is read pixel by pixel. This module draws for the frame window (QGraphicsScene items) and the
«errors over the mask» export (OpenCV, render_marked_errors).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF

from .ui_constants import GRID_INSPECTION_ERROR_TYPE_COLORS, grid_error_display_order

SHAPE_OUTLINE = "outline"
SHAPE_MERGE = "merge"
SHAPE_RING = "ring"
SHAPE_BRACKETS = "brackets"
SHAPE_HATCH = "hatch"
SHAPE_DOTTED = "dotted"
SHAPE_ZONE = "zone"

TYPE_SHAPES: dict[str, str] = {
    "broken_geometry": SHAPE_OUTLINE,
    "merged_contour": SHAPE_MERGE,
    "small_artifact": SHAPE_RING,
    "sparse_cell": SHAPE_BRACKETS,
    "class_conflict": SHAPE_HATCH,
    "network_only": SHAPE_DOTTED,
    "conductor_zone": SHAPE_ZONE,
}
# Shapes drawn first are under the later ones: background, fills, then lines.
_SHAPE_ORDER = (SHAPE_ZONE, SHAPE_HATCH, SHAPE_OUTLINE, SHAPE_MERGE, SHAPE_DOTTED, SHAPE_RING, SHAPE_BRACKETS)
FALLBACK_COLOR = "#94a3b8"


def shape_of(error_type: str) -> str:
    return TYPE_SHAPES.get(str(error_type), SHAPE_OUTLINE)


def color_of(error_type: str) -> str:
    return GRID_INSPECTION_ERROR_TYPE_COLORS.get(str(error_type), FALLBACK_COLOR)


def shown_types(reasons, enabled=None) -> list[str]:
    """The reasons of a cell that are marked, in the filter's order (unknown reasons of older runs last)."""

    present = [str(reason) for reason in reasons or () if str(reason)]
    if enabled is not None:
        allowed = {str(item) for item in enabled}
        present = [reason for reason in present if reason in allowed]
    order = {name: index for index, name in enumerate(grid_error_display_order())}
    return sorted(dict.fromkeys(present), key=lambda reason: (order.get(reason, len(order)), reason))


def _cell_rect(cell) -> tuple[float, float, float, float]:
    x, y, w, h = (float(value) for value in tuple(getattr(cell, "bbox", (0, 0, 1, 1)))[:4])
    return x, y, max(1.0, w), max(1.0, h)


def _outline_points(cell) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in (getattr(cell, "outline", ()) or ())]


# Frame window ---------------------------------------------------------------------------------------


def add_mark_items(scene, cell, scale_x: float, scale_y: float, types, *, dim: bool = False, hot: bool = False) -> list:
    """Scene items marking ``cell`` with these types; ``dim``: faded (another type is singled out),
    ``hot``: under the cursor or chosen (thicker lines and a light fill)."""

    items: list = []
    x, y, w, h = _cell_rect(cell)
    rect = QRectF(x * scale_x, y * scale_y, w * scale_x, h * scale_y)
    points = _outline_points(cell)
    polygon = QPolygonF([QPointF(px * scale_x, py * scale_y) for px, py in points]) if len(points) >= 3 else None
    alpha = 60 if dim else 255
    width = 3.0 if hot else 2.0
    unit = max(scale_x, scale_y)

    def pen(color: str, style=Qt.PenStyle.SolidLine, line_width: float = width) -> QPen:
        value = QColor(color)
        value.setAlpha(alpha)
        result = QPen(value, line_width, style)
        result.setCosmetic(True)
        result.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        return result

    def outline(color: str, style=Qt.PenStyle.SolidLine) -> None:
        if polygon is not None:
            item = scene.addPolygon(polygon, pen(color, style), QBrush(Qt.BrushStyle.NoBrush))
        else:
            item = scene.addRect(rect.adjusted(-unit, -unit, unit, unit), pen(color, style), QBrush(Qt.BrushStyle.NoBrush))
        items.append(item)

    kinds = sorted({shape_of(kind): kind for kind in types}.items(), key=lambda pair: _SHAPE_ORDER.index(pair[0]))
    if hot and not dim and types:
        fill = QColor(color_of(types[0]))
        fill.setAlpha(55)
        items.append(scene.addRect(rect, QPen(Qt.PenStyle.NoPen), QBrush(fill)))
    for shape, kind in kinds:
        color = color_of(kind)
        if shape == SHAPE_ZONE:
            fill = QColor(color)
            fill.setAlpha(min(alpha, 70))
            zone_pen = pen(color, Qt.PenStyle.DashLine, 1.0)
            target = polygon if polygon is not None else None
            item = (
                scene.addPolygon(target, zone_pen, QBrush(fill, Qt.BrushStyle.BDiagPattern))
                if target is not None
                else scene.addRect(rect, zone_pen, QBrush(fill, Qt.BrushStyle.BDiagPattern))
            )
            items.append(item)
        elif shape == SHAPE_HATCH:
            # Diagonal lines drawn as a path: a pattern brush does not follow the zoom of the scene.
            step = max(2.0, min(rect.width(), rect.height()) / 4.0)
            path = QPainterPath()
            offset = -rect.height()
            while offset < rect.width():
                x0, y0 = rect.left() + offset, rect.bottom()
                x1, y1 = x0 + rect.height(), rect.top()
                if x0 < rect.left():
                    y0 -= rect.left() - x0
                    x0 = rect.left()
                if x1 > rect.right():
                    y1 += x1 - rect.right()
                    x1 = rect.right()
                if x1 > x0:
                    path.moveTo(QPointF(x0, y0))
                    path.lineTo(QPointF(x1, y1))
                offset += step
            items.append(scene.addPath(path, pen(color, line_width=1.5)))
            items.append(scene.addRect(rect, pen(color, line_width=1.5), QBrush(Qt.BrushStyle.NoBrush)))
        elif shape == SHAPE_OUTLINE:
            outline(color)
        elif shape == SHAPE_DOTTED:
            outline(color, Qt.PenStyle.DotLine)
        elif shape == SHAPE_MERGE:
            outline(color)
            bar_y = rect.bottom() + 4.0 * unit
            path = QPainterPath(QPointF(rect.left() + rect.width() * 0.15, bar_y))
            path.lineTo(QPointF(rect.right() - rect.width() * 0.15, bar_y))
            items.append(scene.addPath(path, pen(color, line_width=width + 1.0)))
        elif shape == SHAPE_RING:
            radius = max(rect.width(), rect.height()) / 2.0 + 5.0 * unit
            centre = rect.center()
            items.append(
                scene.addEllipse(
                    QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius),
                    pen(color),
                    QBrush(Qt.BrushStyle.NoBrush),
                )
            )
        elif shape == SHAPE_BRACKETS:
            gap = 3.0 * unit
            outer = rect.adjusted(-gap, -gap, gap, gap)
            arm = max(3.0 * unit, min(outer.width(), outer.height()) * 0.35)
            path = QPainterPath()
            for corner, dx, dy in (
                (outer.topLeft(), 1, 1),
                (outer.topRight(), -1, 1),
                (outer.bottomLeft(), 1, -1),
                (outer.bottomRight(), -1, -1),
            ):
                path.moveTo(QPointF(corner.x() + dx * arm, corner.y()))
                path.lineTo(corner)
                path.lineTo(QPointF(corner.x(), corner.y() + dy * arm))
            items.append(scene.addPath(path, pen(color, line_width=width + 0.5)))
    return items


# Export: errors over the mask --------------------------------------------------------------------


def _bgr(color: str) -> tuple[int, int, int]:
    value = QColor(color)
    return int(value.blue()), int(value.green()), int(value.red())


def render_marked_errors(mask: np.ndarray, result, selected_types) -> np.ndarray:
    """The mask in grey with every defect of ``selected_types`` marked by its shape, for a person."""

    import cv2

    gray = np.asarray(mask)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    base = (gray.astype(np.float32) * 0.45).astype(np.uint8)
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    selected = {str(item) for item in selected_types or ()}
    height, width = image.shape[:2]
    for cell in getattr(result, "per_cell_results", ()) or ():
        if str(getattr(cell, "status", "") or "") == "normal":
            continue
        types = shown_types(getattr(cell, "reasons", ()) or (), selected or None)
        if not types:
            continue
        x, y, w, h = (int(round(value)) for value in _cell_rect(cell))
        points = np.asarray(_outline_points(cell), np.int32).reshape(-1, 1, 2)
        for shape, kind in sorted({shape_of(k): k for k in types}.items(), key=lambda pair: _SHAPE_ORDER.index(pair[0])):
            color = _bgr(color_of(kind))
            if shape in (SHAPE_HATCH, SHAPE_ZONE):
                step = 5 if shape == SHAPE_HATCH else 8
                sub = image[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]
                hatch = np.zeros(sub.shape[:2], np.uint8)
                for offset in range(-sub.shape[0], sub.shape[1], step):
                    cv2.line(hatch, (offset, sub.shape[0]), (offset + sub.shape[0], 0), 255, 1)
                sub[hatch > 0] = color
                cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), color, 1)
            elif shape in (SHAPE_OUTLINE, SHAPE_MERGE):
                if len(points) >= 3:
                    cv2.polylines(image, [points], True, color, 2)
                else:
                    cv2.rectangle(image, (x - 1, y - 1), (x + w, y + h), color, 2)
                if shape == SHAPE_MERGE:
                    cv2.line(image, (x + w // 6, y + h + 4), (x + w - w // 6, y + h + 4), color, 3)
            elif shape == SHAPE_DOTTED:
                corners = [(x - 1, y - 1), (x + w, y - 1), (x + w, y + h), (x - 1, y + h)]
                for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
                    length = max(abs(bx - ax), abs(by - ay))
                    for t in range(0, length, 6):
                        px = ax + (bx - ax) * t // max(1, length)
                        py = ay + (by - ay) * t // max(1, length)
                        cv2.circle(image, (int(px), int(py)), 1, color, -1)
            elif shape == SHAPE_RING:
                radius = int(round(max(w, h) / 2 + 5))
                cv2.circle(image, (x + w // 2, y + h // 2), radius, color, 2, cv2.LINE_AA)
            elif shape == SHAPE_BRACKETS:
                gap = 3
                x0, y0, x1, y1 = x - gap, y - gap, x + w - 1 + gap, y + h - 1 + gap
                arm = max(3, int(min(x1 - x0, y1 - y0) * 0.35))
                for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    cv2.line(image, (cx, cy), (cx + dx * arm, cy), color, 2)
                    cv2.line(image, (cx, cy), (cx, cy + dy * arm), color, 2)
    return image
