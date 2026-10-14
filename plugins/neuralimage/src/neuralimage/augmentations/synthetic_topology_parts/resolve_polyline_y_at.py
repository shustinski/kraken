"""synthetic_topology: resolve polyline y at."""
from __future__ import annotations


class ResolvePolylineYAtMixin:
    @staticmethod
    def _resolve_polyline_y_at_x(points: list[tuple[int, int]], x: int) -> int:
        if not points:
            return 0
        resolved_x = int(x)
        nearest_y = points[0][1]
        nearest_distance = abs(points[0][0] - resolved_x)
        for start_point, end_point in zip(points[:-1], points[1:]):
            if start_point[1] == end_point[1]:
                segment_min_x = min(start_point[0], end_point[0])
                segment_max_x = max(start_point[0], end_point[0])
                if segment_min_x <= resolved_x <= segment_max_x:
                    return int(start_point[1])
            for point_x, point_y in (start_point, end_point):
                distance = abs(point_x - resolved_x)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_y = point_y
        return int(nearest_y)
