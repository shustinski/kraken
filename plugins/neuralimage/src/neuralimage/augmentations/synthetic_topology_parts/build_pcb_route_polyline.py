"""synthetic_topology: build pcb route polyline."""
from __future__ import annotations
import random
import cv2
import numpy as np


class BuildPcbRoutePolylineMixin:
    def _build_pcb_route_component(
        self,
        height: int,
        width: int,
        rng: random.Random,
        *,
        occupied_guard_mask: np.ndarray,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        for _attempt_index in range(96):
            segment_count, trace_half_width = self._sample_trace_shape(rng)
            polyline = self._build_pcb_route_polyline(height, width, rng, segment_count=segment_count)
            if len(polyline) < 2:
                continue
            candidate_mask = np.zeros((height, width), dtype=np.uint8)
            self._draw_polyline(candidate_mask, polyline, thickness=trace_half_width * 2 + 1)
            if int(np.count_nonzero(candidate_mask)) <= 0:
                continue
            if np.any((candidate_mask > 0) & (occupied_guard_mask > 0)):
                continue
            anchors = [polyline[0], polyline[-1]]
            if len(polyline) > 2:
                anchors.extend(polyline[1:-1])
            anchors.append(self._midpoint(polyline[0], polyline[-1]))
            return candidate_mask, anchors
        return None, []

    def _build_pcb_route_polyline(
        self,
        height: int,
        width: int,
        rng: random.Random,
        *,
        segment_count: int,
    ) -> list[tuple[int, int]]:
        margin = self._effective_margin(height, width)
        horizontal_route = bool(rng.getrandbits(1))
        if segment_count <= 1:
            orientation = str(rng.choices(
                ('horizontal', 'vertical', 'diag_pos', 'diag_neg'),
                weights=(0.3, 0.18, 0.26, 0.26),
                k=1,
            )[0])
            segment = self._full_span_segment(height, width, orientation, self._random_parallel_offset(height, width, orientation, rng))
            return [] if segment is None else [segment[0], segment[1]]
        if horizontal_route:
            start_y = rng.randint(margin, max(margin + 1, height - margin - 1))
            end_y = rng.randint(margin, max(margin + 1, height - margin - 1))
            start = (0, start_y)
            end = (width - 1, end_y)
            if abs(end_y - start_y) <= 1:
                return [start, end]
            diagonal_span = abs(end_y - start_y)
            if segment_count == 2:
                bend_x = max(0, min(width - 1, (width - 1) - diagonal_span))
                if bend_x <= 0 or bend_x >= width - 1:
                    return [start, end]
                return [start, (bend_x, start_y), end]
            max_first = max(4, (width - 1) - diagonal_span - 4)
            first_x = rng.randint(4, max_first)
            second_x = min(width - 1, first_x + diagonal_span)
            return [start, (first_x, start_y), (second_x, end_y), end]
        start_x = rng.randint(margin, max(margin + 1, width - margin - 1))
        end_x = rng.randint(margin, max(margin + 1, width - margin - 1))
        start = (start_x, 0)
        end = (end_x, height - 1)
        if abs(end_x - start_x) <= 1:
            return [start, end]
        diagonal_span = abs(end_x - start_x)
        if segment_count == 2:
            bend_y = max(0, min(height - 1, (height - 1) - diagonal_span))
            if bend_y <= 0 or bend_y >= height - 1:
                return [start, end]
            return [start, (start_x, bend_y), end]
        max_first = max(4, (height - 1) - diagonal_span - 4)
        first_y = rng.randint(4, max_first)
        second_y = min(height - 1, first_y + diagonal_span)
        return [start, (start_x, first_y), (end_x, second_y), end]

    def _build_pcb_cell_motif(
        self,
        canvas_shape: tuple[int, int],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        rng: random.Random,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        width = max(12, x2 - x1)
        height = max(12, y2 - y1)
        candidate_mask = np.zeros(canvas_shape, dtype=np.uint8)
        trace_half_width = rng.randint(*self._width_range())
        pad = max(3, trace_half_width + 1)
        center_x = x1 + (width // 2)
        center_y = y1 + (height // 2)
        if rng.random() < 0.5:
            start = (x1 + pad, center_y)
            end = (x2 - pad - 1, center_y)
        else:
            start = (center_x, y1 + pad)
            end = (center_x, y2 - pad - 1)
        cv2.line(candidate_mask, start, end, 255, thickness=(trace_half_width * 2 + 1))
        cv2.circle(candidate_mask, start, max(2, trace_half_width + 1), 255, thickness=-1)
        cv2.circle(candidate_mask, end, max(2, trace_half_width + 1), 255, thickness=-1)
        return candidate_mask, [start, end, self._midpoint(start, end)]
