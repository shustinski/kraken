"""synthetic_topology: segments intersect."""
from __future__ import annotations
import random
import cv2
import numpy as np


class SegmentsIntersectMixin:
    def _sample_trace_shape(self, rng: random.Random) -> tuple[int, int]:
        min_segments, max_segments = self._segment_range()
        min_width, max_width = self._width_range()
        return (
            rng.randint(min_segments, max_segments),
            rng.randint(min_width, max_width),
        )

    def _segment_range(self) -> tuple[int, int]:
        segment_range = self.params.segment_count_range or (self.params.segment_count, self.params.segment_count)
        min_segments, max_segments = sorted((max(1, int(segment_range[0])), max(1, int(segment_range[1]))))
        return min_segments, max_segments

    def _width_range(self) -> tuple[int, int]:
        width_range = self.params.trace_half_width_range or (self.params.trace_half_width, self.params.trace_half_width)
        min_width, max_width = sorted((max(1, int(width_range[0])), max(1, int(width_range[1]))))
        return min_width, max_width

    def _effective_margin(self, height: int, width: int) -> int:
        return min(
            max(4, int(self.params.margin)),
            max(4, height // 4),
            max(4, width // 4),
        )

    @staticmethod
    def _segment_intersects_existing(
        candidate_segment: tuple[tuple[int, int], tuple[int, int]],
        segments: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> bool:
        from neuralimage.augmentations.synthetic_topology import SyntheticTopologyGenerator

        if not segments:
            return False
        candidate_start, candidate_end = candidate_segment
        for index, existing_segment in enumerate(segments):
            existing_start, existing_end = existing_segment
            if not SyntheticTopologyGenerator._segments_intersect(
                candidate_start,
                candidate_end,
                existing_start,
                existing_end,
            ):
                continue
            is_last_segment = index == (len(segments) - 1)
            if (
                is_last_segment
                and existing_end == candidate_start
                and SyntheticTopologyGenerator._forms_only_corner(existing_segment, candidate_segment)
            ):
                continue
            return True
        return False

    @staticmethod
    def _forms_only_corner(
        existing_segment: tuple[tuple[int, int], tuple[int, int]],
        candidate_segment: tuple[tuple[int, int], tuple[int, int]],
    ) -> bool:
        existing_start, existing_end = existing_segment
        candidate_start, candidate_end = candidate_segment
        if existing_end != candidate_start:
            return False
        existing_horizontal = existing_start[1] == existing_end[1]
        candidate_horizontal = candidate_start[1] == candidate_end[1]
        return existing_horizontal != candidate_horizontal

    @staticmethod
    def _segments_intersect(
        start_a: tuple[int, int],
        end_a: tuple[int, int],
        start_b: tuple[int, int],
        end_b: tuple[int, int],
    ) -> bool:
        ax1, ay1 = start_a
        ax2, ay2 = end_a
        bx1, by1 = start_b
        bx2, by2 = end_b

        a_horizontal = ay1 == ay2
        b_horizontal = by1 == by2
        if a_horizontal and b_horizontal:
            if ay1 != by1:
                return False
            a_min_x, a_max_x = sorted((ax1, ax2))
            b_min_x, b_max_x = sorted((bx1, bx2))
            return max(a_min_x, b_min_x) <= min(a_max_x, b_max_x)
        if (not a_horizontal) and (not b_horizontal):
            if ax1 != bx1:
                return False
            a_min_y, a_max_y = sorted((ay1, ay2))
            b_min_y, b_max_y = sorted((by1, by2))
            return max(a_min_y, b_min_y) <= min(a_max_y, b_max_y)

        if a_horizontal:
            h_start, h_end = start_a, end_a
            v_start, v_end = start_b, end_b
        else:
            h_start, h_end = start_b, end_b
            v_start, v_end = start_a, end_a
        hx1, hy = h_start
        hx2, _hy2 = h_end
        vx, vy1 = v_start
        _vx2, vy2 = v_end
        h_min_x, h_max_x = sorted((hx1, hx2))
        v_min_y, v_max_y = sorted((vy1, vy2))
        return h_min_x <= vx <= h_max_x and v_min_y <= hy <= v_max_y

    def _build_guard_mask(self, mask_u8: np.ndarray, *, radius: int | None = None) -> np.ndarray:
        clearance = max(0, int(self.params.trace_clearance))
        if clearance <= 0:
            return (mask_u8 > 0).astype(np.uint8) * 255
        resolved_radius = max(1, int(radius) if radius is not None else (self._width_range()[1] + clearance))
        kernel_size = resolved_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask_u8, kernel)

    def _mask_overlaps_guard(
        self,
        candidate_mask: np.ndarray,
        existing_mask: np.ndarray,
        *,
        allowed_points: list[tuple[int, int]] | None = None,
        allowance_radius: int = 0,
        radius: int | None = None,
    ) -> bool:
        if int(np.count_nonzero(candidate_mask)) <= 0 or int(np.count_nonzero(existing_mask)) <= 0:
            return False
        guard_mask = self._build_guard_mask(existing_mask, radius=radius).copy()
        if allowed_points:
            resolved_allowance_radius = max(0, int(allowance_radius))
            for point_x, point_y in allowed_points:
                if resolved_allowance_radius <= 0:
                    if 0 <= int(point_x) < guard_mask.shape[1] and 0 <= int(point_y) < guard_mask.shape[0]:
                        guard_mask[int(point_y), int(point_x)] = 0
                    continue
                cv2.circle(guard_mask, (int(point_x), int(point_y)), resolved_allowance_radius, 0, thickness=-1)
        return bool(np.any((candidate_mask > 0) & (guard_mask > 0)))

    @staticmethod
    def _draw_polyline(canvas: np.ndarray, points: list[tuple[int, int]], *, thickness: int) -> None:
        if len(points) < 2:
            return
        resolved_thickness = max(1, int(thickness))
        cv2.polylines(
            canvas,
            [np.asarray(points, dtype=np.int32)],
            isClosed=False,
            color=255,
            thickness=resolved_thickness,
            lineType=cv2.LINE_8,
        )

    def _build_fallback_mask(self, height: int, width: int) -> np.ndarray:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        center_y = height // 2
        center_x = width // 2
        fallback_half_span = max(10, min(height, width) // 8)
        cv2.line(
            mask_u8,
            (max(0, center_x - fallback_half_span), center_y),
            (min(width - 1, center_x + fallback_half_span), center_y),
            255,
            thickness=max(1, self._width_range()[1] * 2 + 1),
        )
        cv2.circle(mask_u8, (max(0, center_x - fallback_half_span), center_y), max(3, self._width_range()[1] + 1), 255, thickness=-1)
        cv2.circle(mask_u8, (min(width - 1, center_x + fallback_half_span), center_y), max(3, self._width_range()[1] + 1), 255, thickness=-1)
        return mask_u8

    def _render_image(self, mask_u8: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
        if self._domain() == 'pcb':
            return self._render_pcb_rgb(mask_u8, np_rng)
        return self._render_ic_grayscale(mask_u8, np_rng)
