"""scene_renderer: draw mos contrast overlay."""
from __future__ import annotations
import math
from PyQt6.QtGui import QBrush, QColor, QPen
from krona.infrastructure.edif_parser import EdifTextParser


class DrawMosContrastOverlayMixin:
    def _draw_mos_contrast_overlay(self, draw_block: str, transform: Affine2D) -> None:
        light_pen = QPen(QColor(214, 230, 255, 170))
        light_pen.setWidthF(1.2)
        light_pen.setCosmetic(True)
        for figure_block in EdifTextParser.find_direct_classes(draw_block, "(figure "):
            for path_block in EdifTextParser.find_classes(figure_block, "(path"):
                for point_list_block in EdifTextParser.find_classes(path_block, "(pointList"):
                    points = EdifTextParser.parse_points(point_list_block)
                    if len(points) < 2:
                        continue
                    world = [transform.apply(p.x, p.y) for p in points]
                    for start, end in zip(world, world[1:]):
                        x1, y1 = self._to_scene(start[0], start[1])
                        x2, y2 = self._to_scene(end[0], end[1])
                        item = self._scene.addLine(x1, y1, x2, y2, light_pen)
                        item.setZValue(905)

            for arc_block in EdifTextParser.find_classes(figure_block, "(arc"):
                points = EdifTextParser.parse_points(arc_block)
                if len(points) < 3:
                    continue
                world = [
                    transform.apply(points[0].x, points[0].y),
                    transform.apply(points[1].x, points[1].y),
                    transform.apply(points[2].x, points[2].y),
                ]
                self._draw_arc(world, light_pen)

            for rect_block in EdifTextParser.find_classes(figure_block, "(rectangle"):
                points = EdifTextParser.parse_points(rect_block)
                if len(points) < 2:
                    continue
                p1 = transform.apply(points[0].x, points[0].y)
                p2 = transform.apply(points[1].x, points[1].y)
                x1, y1 = self._to_scene(p1[0], p1[1])
                x2, y2 = self._to_scene(p2[0], p2[1])
                item = self._scene.addRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), light_pen)
                item.setZValue(905)

            for circle_block in EdifTextParser.find_classes(figure_block, "(circle"):
                points = EdifTextParser.parse_points(circle_block)
                if len(points) < 2:
                    continue
                center = transform.apply(points[0].x, points[0].y)
                edge = transform.apply(points[1].x, points[1].y)
                radius = math.hypot(edge[0] - center[0], edge[1] - center[1])
                cx, cy = self._to_scene(center[0], center[1])
                item = self._scene.addEllipse(
                    cx - radius * self._scale,
                    cy - radius * self._scale,
                    2 * radius * self._scale,
                    2 * radius * self._scale,
                    light_pen,
                )
                item.setZValue(905)

            for dot_block in EdifTextParser.find_classes(figure_block, "(dot"):
                points = EdifTextParser.parse_points(dot_block)
                if not points:
                    continue
                p = transform.apply(points[0].x, points[0].y)
                x, y = self._to_scene(p[0], p[1])
                item = self._scene.addEllipse(x - 1.8, y - 1.8, 3.6, 3.6, light_pen, QBrush(light_pen.color()))
                item.setZValue(905)
