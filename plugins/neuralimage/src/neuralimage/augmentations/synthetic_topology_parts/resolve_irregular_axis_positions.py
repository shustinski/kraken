"""synthetic_topology: resolve irregular axis positions."""
from __future__ import annotations
import math
import random
import cv2
import numpy as np


class ResolveIrregularAxisPositionsMixin:
    def _build_ic_cell_motif(
        self,
        canvas_shape: tuple[int, int],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        rng: random.Random,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        cell_w = max(14, x2 - x1)
        cell_h = max(14, y2 - y1)
        trace_half_width = rng.randint(*self._width_range())
        candidate_mask = np.zeros(canvas_shape, dtype=np.uint8)
        inset_x = max(4, trace_half_width + 2)
        inset_y = max(3, trace_half_width + 1)
        if (x2 - x1) < 12 or (y2 - y1) < 10:
            return None, []
        band_mask, band_anchors = self._build_ic_serpentine_rect(
            canvas_shape,
            x1 + inset_x,
            y1 + inset_y,
            x2 - inset_x,
            y2 - inset_y,
            trace_half_width,
            rng,
            density_scale=0.85,
        )
        if band_mask is None:
            return None, []
        candidate_mask = np.maximum(candidate_mask, band_mask)
        mid_y = y1 + int(round(cell_h * 0.5))
        left_anchor = (max(0, x1), int(np.clip(mid_y, 0, canvas_shape[0] - 1)))
        right_anchor = (min(canvas_shape[1] - 1, x2 - 1), int(np.clip(mid_y, 0, canvas_shape[0] - 1)))
        cv2.line(candidate_mask, left_anchor, (x1 + inset_x, left_anchor[1]), 255, thickness=(trace_half_width * 2 + 1))
        cv2.line(candidate_mask, (x2 - inset_x - 1, right_anchor[1]), right_anchor, 255, thickness=(trace_half_width * 2 + 1))
        return candidate_mask, [left_anchor, right_anchor, *band_anchors]

    def _resolve_neighbor_stripe_bounds(
        self,
        centers: list[int],
        index: int,
        extent: int,
    ) -> tuple[int, int]:
        center = int(centers[index])
        prev_center = int(centers[index - 1]) if index > 0 else 0
        next_center = int(centers[index + 1]) if index < len(centers) - 1 else (extent - 1)
        stripe_min = int(max(0, round((prev_center + center) / 2.0)))
        stripe_max = int(min(extent - 1, round((next_center + center) / 2.0)))
        return stripe_min, stripe_max

    def _resolve_irregular_axis_positions(
        self,
        *,
        axis_extent: int,
        spacing: int,
        requested: int,
        rng: random.Random,
    ) -> list[int]:
        margin = min(max(4, int(self.params.margin)), max(4, axis_extent // 4))
        usable_min = int(margin)
        usable_max = int(max(usable_min + 1, axis_extent - margin - 1))
        usable_span = max(1, usable_max - usable_min)
        max_count = max(1, 1 + (usable_span // max(1, spacing)))
        count = max(1, min(int(requested), max_count))
        if count <= 1:
            return [rng.randint(usable_min, usable_max)]
        pitch = usable_span / float(max(1, count - 1))
        jitter_radius = max(1, int(round(min(pitch * 0.28, spacing * 0.35))))
        positions: list[int] = []
        for index in range(count):
            base = usable_min + (pitch * float(index))
            if index == 0:
                position = usable_min + rng.randint(0, jitter_radius)
            elif index == count - 1:
                position = usable_max - rng.randint(0, jitter_radius)
            else:
                position = int(round(base)) + rng.randint(-jitter_radius, jitter_radius)
            positions.append(int(np.clip(position, usable_min, usable_max)))
        for index in range(1, len(positions)):
            minimum = positions[index - 1] + max(1, spacing)
            positions[index] = max(positions[index], minimum)
        for index in range(len(positions) - 2, -1, -1):
            maximum = positions[index + 1] - max(1, spacing)
            positions[index] = min(positions[index], maximum)
        sanitized = [
            int(np.clip(position, usable_min, usable_max))
            for position in positions
        ]
        if not sanitized:
            return [rng.randint(usable_min, usable_max)]
        return sanitized

    def _build_ic_routing_profile(self, height: int, width: int) -> dict[str, int]:
        min_width, max_width = self._width_range()
        trace_half_width = max(1, int(round((min_width + max_width) / 2.0)))
        track_step = max(4, (trace_half_width * 3) + max(2, int(self.params.trace_clearance)))
        x_margin = max(6, min(width // 8, int(self.params.margin) + trace_half_width + 2))
        y_margin = max(6, min(height // 8, int(self.params.margin) + trace_half_width + 2))
        return {
            'trace_half_width': trace_half_width,
            'track_step': track_step,
            'x_margin': x_margin,
            'y_margin': y_margin,
        }

    def _create_ic_routing_state(self, height: int, width: int, profile: dict[str, int]) -> dict[str, object]:
        xs = [0] + list(range(profile['x_margin'], max(profile['x_margin'] + 1, width - profile['x_margin']), profile['track_step'])) + [width - 1]
        ys = [0] + list(range(profile['y_margin'], max(profile['y_margin'] + 1, height - profile['y_margin']), profile['track_step'])) + [height - 1]
        xs = sorted(set(int(value) for value in xs))
        ys = sorted(set(int(value) for value in ys))
        if len(xs) < 10:
            xs = sorted(set([0, width - 1] + list(np.linspace(profile['x_margin'], width - profile['x_margin'] - 1, 10, dtype=int))))
        if len(ys) < 8:
            ys = sorted(set([0, height - 1] + list(np.linspace(profile['y_margin'], height - profile['y_margin'] - 1, 8, dtype=int))))
        grid_height = len(ys)
        grid_width = len(xs)
        return {
            'mask': np.zeros((height, width), dtype=np.uint8),
            'blocked': np.zeros((grid_height, grid_width), dtype=np.uint8),
            'xs': xs,
            'ys': ys,
            'grid_width': grid_width,
            'grid_height': grid_height,
            'profile': profile,
        }

    @staticmethod
    def _nearest_grid_y(pixel_y: int, ys: list[int]) -> int:
        distances = [abs(int(pixel_y) - int(candidate)) for candidate in ys]
        return int(distances.index(min(distances)))

    @staticmethod
    def _randint_safe(rng: random.Random, low: int, high: int) -> int:
        resolved_low = int(min(low, high))
        resolved_high = int(max(low, high))
        return rng.randint(resolved_low, resolved_high)

    def _route_ic_nets(
        self,
        routing_state: dict[str, object],
        nets: list[tuple[tuple[int, int], tuple[int, int]]],
        rng: random.Random,
        *,
        family: str,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        anchors: list[tuple[int, int]] = []
        blocked = np.asarray(routing_state['blocked'], dtype=np.uint8)
        mask_u8 = np.asarray(routing_state['mask'], dtype=np.uint8)
        base_padding = 0 if family in ('channels', 'cell_array') else 1
        for start, end in nets:
            if blocked[start[1], start[0]] > 0 or blocked[end[1], end[0]] > 0:
                continue
            path = self._pattern_route(start, end, blocked, family=family, rng=rng)
            if path is None:
                path = self._maze_route(start, end, blocked, family=family, rng=rng)
            if not path or len(path) < 2:
                continue
            route_half_width = self._sample_ic_route_half_width(rng)
            candidate_mask = self._build_grid_path_mask(
                mask_shape=mask_u8.shape,
                path=path,
                routing_state=routing_state,
                thickness=(route_half_width * 2 + 1),
            )
            if self._mask_overlaps_guard(
                candidate_mask,
                mask_u8,
                radius=max(1, int(self.params.trace_clearance)),
            ):
                continue
            route_padding = max(base_padding, self._grid_padding_for_half_width(routing_state, route_half_width))
            self._occupy_grid_path(blocked, path, padding=route_padding)
            mask_u8[:] = np.maximum(mask_u8, candidate_mask)
            anchors.extend(self._grid_path_to_pixels(path, routing_state))
        return mask_u8, anchors

    def _sample_ic_route_half_width(self, rng: random.Random) -> int:
        min_width, max_width = self._width_range()
        return int(rng.randint(min_width, max_width))

    def _grid_padding_for_half_width(self, routing_state: dict[str, object], trace_half_width: int) -> int:
        xs = list(routing_state['xs'])
        ys = list(routing_state['ys'])
        x_steps = [abs(xs[index + 1] - xs[index]) for index in range(len(xs) - 1) if abs(xs[index + 1] - xs[index]) > 0]
        y_steps = [abs(ys[index + 1] - ys[index]) for index in range(len(ys) - 1) if abs(ys[index + 1] - ys[index]) > 0]
        grid_step = max(1, min(x_steps + y_steps) if (x_steps or y_steps) else 1)
        return max(0, int(math.ceil((max(1, int(trace_half_width)) + max(0, int(self.params.trace_clearance))) / float(grid_step))) - 1)
