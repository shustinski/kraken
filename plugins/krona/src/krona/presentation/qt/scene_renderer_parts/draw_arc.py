"""scene_renderer: draw arc."""
from __future__ import annotations
import math
from PyQt6.QtGui import QPen


class DrawArcMixin:
    def _draw_arc(self, points: list[tuple[float, float]], pen: QPen, instance_name: str | None = None) -> None:
        if len(points) < 3:
            self._draw_polyline(points, pen, instance_name=instance_name)
            return
        s1, s2, s3 = (self._to_scene(points[0][0], points[0][1]), self._to_scene(points[1][0], points[1][1]), self._to_scene(points[2][0], points[2][1]))
        x1, y1 = s1
        x2, y2 = s2
        x3, y3 = s3
        det = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(det) < 1e-6:
            self._draw_polyline([s1, s2, s3], pen, instance_name=instance_name)
            return
        ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / det
        uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / det
        radius = math.hypot(x1 - ux, y1 - uy)
        if radius < 0.2:
            return
        a1 = math.atan2(y1 - uy, x1 - ux)
        a2 = math.atan2(y2 - uy, x2 - ux)
        a3 = math.atan2(y3 - uy, x3 - ux)

        def normalize(angle: float) -> float:
            while angle < 0:
                angle += 2 * math.pi
            while angle >= 2 * math.pi:
                angle -= 2 * math.pi
            return angle

        def mid_on_ccw(start: float, end: float, mid: float) -> bool:
            start = normalize(start)
            end = normalize(end)
            mid = normalize(mid)
            if end < start:
                end += 2 * math.pi
            if mid < start:
                mid += 2 * math.pi
            return start <= mid <= end

        sweep = normalize(a3) - normalize(a1)
        if sweep < 0:
            sweep += 2 * math.pi
        if not mid_on_ccw(a1, a1 + sweep, a2):
            sweep -= 2 * math.pi
        steps = max(10, int(abs(sweep) * radius / 10.0))
        pts: list[tuple[float, float]] = []
        for idx in range(steps + 1):
            angle = a1 + sweep * (idx / steps)
            pts.append((ux + radius * math.cos(angle), uy + radius * math.sin(angle)))
        for start, end in zip(pts, pts[1:]):
            item = self._scene.addLine(start[0], start[1], end[0], end[1], pen)
            self._tag_item(item, instance_name)
            self._stats.wire_segment_count += 1
