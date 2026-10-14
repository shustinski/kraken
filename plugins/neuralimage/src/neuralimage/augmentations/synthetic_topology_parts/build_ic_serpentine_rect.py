"""synthetic_topology: build ic serpentine rect."""
from __future__ import annotations
import random
import cv2
import numpy as np


class BuildIcSerpentineRectMixin:
    def _build_ic_serpentine_band_between_trunks(
        self,
        canvas_shape: tuple[int, int],
        start_a: tuple[int, int],
        end_a: tuple[int, int],
        start_b: tuple[int, int],
        end_b: tuple[int, int],
        trace_half_width: int,
        rng: random.Random,
        *,
        density_scale: float = 1.0,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        height, width = canvas_shape
        if start_a[1] == end_a[1] and start_b[1] == end_b[1]:
            y_top = min(start_a[1], start_b[1])
            y_bottom = max(start_a[1], start_b[1])
            return self._build_ic_serpentine_rect(
                canvas_shape,
                0,
                y_top + trace_half_width + 2,
                width - 1,
                y_bottom - trace_half_width - 2,
                trace_half_width,
                rng,
                density_scale=density_scale,
            )
        if start_a[0] == end_a[0] and start_b[0] == end_b[0]:
            x_left = min(start_a[0], start_b[0])
            x_right = max(start_a[0], start_b[0])
            rotated_mask, rotated_anchors = self._build_ic_serpentine_rect(
                (width, height),
                0,
                x_left + trace_half_width + 2,
                height - 1,
                x_right - trace_half_width - 2,
                trace_half_width,
                rng,
                density_scale=density_scale,
            )
            if rotated_mask is None:
                return None, []
            mask = np.rot90(rotated_mask, k=3)
            anchors = [(int(y), int(width - 1 - x)) for x, y in rotated_anchors]
            return mask, anchors
        return None, []

    def _build_ic_serpentine_rect(
        self,
        canvas_shape: tuple[int, int],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        trace_half_width: int,
        rng: random.Random,
        *,
        density_scale: float = 1.0,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        left = int(min(x1, x2))
        right = int(max(x1, x2))
        top = int(min(y1, y2))
        bottom = int(max(y1, y2))
        if (right - left) < 14 or (bottom - top) < 10:
            return None, []
        candidate_mask = np.zeros(canvas_shape, dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        thickness = trace_half_width * 2 + 1
        lane_pitch = max(4, thickness + max(2, int(self.params.trace_clearance)) + 1)
        lane_count = max(2, int(((bottom - top) / max(1, lane_pitch)) * float(density_scale)))
        lane_count = min(lane_count, max(2, (bottom - top) // max(3, lane_pitch)))
        if lane_count < 2:
            lane_count = 2
        lane_positions = np.linspace(top + lane_pitch // 2, bottom - lane_pitch // 2, lane_count)
        left_pad = left + max(2, thickness)
        right_pad = right - max(2, thickness)
        if right_pad - left_pad < 8:
            return None, []
        base_shift = max(4, int((right_pad - left_pad) * 0.12))
        for lane_index, lane_pos in enumerate(lane_positions):
            y = int(round(float(lane_pos)))
            shift = int(round(base_shift * rng.uniform(0.7, 1.3)))
            start_x = left_pad + (shift if lane_index % 2 else 0)
            end_x = right_pad - (0 if lane_index % 2 else shift)
            if end_x - start_x < 8:
                start_x = left_pad
                end_x = right_pad
            cv2.line(candidate_mask, (start_x, y), (end_x, y), 255, thickness=thickness)
            anchors.extend(((start_x, y), (end_x, y)))
            if lane_index >= len(lane_positions) - 1:
                continue
            next_y = int(round(float(lane_positions[lane_index + 1])))
            connector_x = end_x if lane_index % 2 == 0 else start_x
            cv2.line(candidate_mask, (connector_x, y), (connector_x, next_y), 255, thickness=thickness)
            anchors.append((connector_x, next_y))
            if (lane_index % 2 == 0 and rng.random() < 0.55) or (lane_index % 2 == 1 and rng.random() < 0.42):
                jog_x = int(np.clip(
                    connector_x + ((-1 if lane_index % 2 == 0 else 1) * int(round((right_pad - left_pad) * rng.uniform(0.08, 0.18)))),
                    left_pad,
                    right_pad,
                ))
                mid_y = int(round((y + next_y) / 2.0))
                cv2.line(candidate_mask, (connector_x, mid_y), (jog_x, mid_y), 255, thickness=thickness)
                anchors.append((jog_x, mid_y))
        return candidate_mask, anchors
