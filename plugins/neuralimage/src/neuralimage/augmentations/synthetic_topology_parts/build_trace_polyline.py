"""synthetic_topology: build trace polyline."""
from __future__ import annotations
import random
import numpy as np


class BuildTracePolylineMixin:
    def _build_trace_polyline(
        self,
        height: int,
        width: int,
        rng: random.Random,
        *,
        segment_count: int,
    ) -> list[tuple[int, int]]:
        margin = self._effective_margin(height, width)
        min_x = margin
        max_x = max(margin + 1, width - margin - 1)
        min_y = margin
        max_y = max(margin + 1, height - margin - 1)

        current_x = rng.randint(min_x, max_x)
        current_y = rng.randint(min_y, max_y)
        horizontal = bool(rng.getrandbits(1))
        points: list[tuple[int, int]] = [(current_x, current_y)]
        segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        min_span = max(8, min(height, width) // 10)
        max_span = max(min_span + 2, min(height, width) // 2)

        for _segment_index in range(max(1, int(segment_count))):
            next_point: tuple[int, int] | None = None
            for _branch_attempt in range(16):
                segment_length = rng.randint(min_span, max_span)
                direction = -1 if rng.random() < 0.5 else 1
                next_x = current_x
                next_y = current_y
                if horizontal:
                    next_x = int(np.clip(current_x + direction * segment_length, min_x, max_x))
                else:
                    next_y = int(np.clip(current_y + direction * segment_length, min_y, max_y))
                if (next_x, next_y) == points[-1]:
                    continue
                candidate_segment = ((current_x, current_y), (next_x, next_y))
                if self._segment_intersects_existing(candidate_segment, segments):
                    continue
                next_point = (next_x, next_y)
                break
            if next_point is None:
                break
            points.append(next_point)
            segments.append(((current_x, current_y), next_point))
            current_x, current_y = next_point
            horizontal = not horizontal

        if len(points) == 1:
            fallback_x = int(np.clip(current_x + min_span, min_x, max_x))
            if fallback_x == current_x:
                fallback_x = int(np.clip(current_x - min_span, min_x, max_x))
            if fallback_x != current_x:
                candidate_segment = ((current_x, current_y), (fallback_x, current_y))
                if not self._segment_intersects_existing(candidate_segment, segments):
                    points.append((fallback_x, current_y))
        return points
