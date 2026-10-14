"""scene_renderer: draw polyline."""
from __future__ import annotations
from PyQt6.QtGui import QBrush, QPen


class DrawPolylineMixin:
    def _draw_polyline(self, points: list[tuple[float, float]], pen: QPen, instance_name: str | None = None) -> None:
        if len(points) < 2:
            if points:
                x, y = self._to_scene(points[0][0], points[0][1])
                item = self._scene.addEllipse(x - 1.5, y - 1.5, 3.0, 3.0, pen, QBrush(pen.color()))
                self._tag_item(item, instance_name)
                self._stats.wire_segment_count += 1
            return
        for start, end in zip(points, points[1:]):
            x1, y1 = self._to_scene(start[0], start[1])
            x2, y2 = self._to_scene(end[0], end[1])
            item = self._scene.addLine(x1, y1, x2, y2, pen)
            self._tag_item(item, instance_name)
            self._stats.wire_segment_count += 1
