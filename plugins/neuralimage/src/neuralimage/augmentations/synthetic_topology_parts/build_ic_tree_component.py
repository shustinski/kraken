"""synthetic_topology: build ic tree component."""
from __future__ import annotations
import random
import cv2
import numpy as np


class BuildIcTreeComponentMixin:
    def _build_ic_tree_component(
        self,
        canvas_shape: tuple[int, int],
        row_center: int,
        stripe_min: int,
        stripe_max: int,
        rng: random.Random,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        height, width = canvas_shape
        trace_half_width = rng.randint(*self._width_range())
        row_y = int(np.clip(row_center, stripe_min + trace_half_width + 2, stripe_max - trace_half_width - 2))
        if (stripe_max - stripe_min) < max(12, trace_half_width * 5):
            return None, []
        component_mask = np.zeros(canvas_shape, dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        trunk_points = self._build_ic_tree_backbone_points(
            width,
            row_y,
            stripe_min,
            stripe_max,
            rng,
        )
        self._draw_polyline(component_mask, trunk_points, thickness=(trace_half_width * 2 + 1))
        anchors.extend((trunk_points[0], trunk_points[-1], *trunk_points[1:-1]))
        local_occupied = self._build_guard_mask(component_mask)
        trunk_segments = [
            (start_point, end_point)
            for start_point, end_point in zip(trunk_points[:-1], trunk_points[1:])
            if start_point[1] == end_point[1]
        ]
        if not trunk_segments:
            trunk_segments = [(trunk_points[0], trunk_points[-1])]

        branch_target = max(3, min(7, int(round(self.params.trace_count / 2.0))))
        for branch_index, branch_fraction in enumerate(self._resolve_branch_fractions(branch_target)):
            branch_half_width = rng.randint(*self._width_range())
            selected_segment = trunk_segments[min(len(trunk_segments) - 1, branch_index % len(trunk_segments))]
            branch_start = self._point_on_segment(selected_segment[0], selected_segment[1], branch_fraction)
            upward_space = branch_start[1] - stripe_min - 3
            downward_space = stripe_max - branch_start[1] - 3
            if upward_space < 10 and downward_space < 10:
                continue
            preferred_direction = -1 if (branch_index % 2 == 0) else 1
            candidate_directions = (preferred_direction, -preferred_direction)
            branch_end: tuple[int, int] | None = None
            for direction in candidate_directions:
                max_length = upward_space if direction < 0 else downward_space
                if max_length <= 8:
                    continue
                candidate_end = (
                    int(branch_start[0]),
                    int(np.clip(
                        branch_start[1] + (direction * max(10, int(round(max_length * rng.uniform(0.46, 0.82))))),
                        stripe_min + 1,
                        stripe_max - 1,
                    )),
                )
                branch_mask = np.zeros_like(component_mask)
                cv2.line(branch_mask, branch_start, candidate_end, 255, thickness=(branch_half_width * 2 + 1))
                if not self._allow_candidate_overlap(
                    local_occupied,
                    branch_mask,
                    canvas_shape,
                    [branch_start],
                    allowance_radius=max(2, branch_half_width + 1),
                ):
                    continue
                component_mask = np.maximum(component_mask, branch_mask)
                local_occupied = self._build_guard_mask(component_mask)
                branch_end = candidate_end
                anchors.extend((branch_start, branch_end))
                break
            if branch_end is None:
                continue

            leaf_count = max(2, min(5, 2 + int(rng.random() < 0.7) + int(rng.random() < 0.35)))
            spine_fractions = np.linspace(0.18, 0.88, leaf_count)
            dominant_side = -1 if rng.random() < 0.5 else 1
            for tap_index, tap_fraction in enumerate(spine_fractions):
                tap_start = self._point_on_segment(branch_start, branch_end, float(tap_fraction))
                tap_direction = dominant_side if (tap_index % 3 != 1) else -dominant_side
                tap_length = max(8, int(round(width * rng.uniform(0.06, 0.14))))
                tap_end = self._build_axis_branch_endpoint(
                    tap_start,
                    length=tap_length,
                    direction=tap_direction,
                    axis='x',
                    width=width,
                    height=height,
                )
                tap_half_width = max(1, branch_half_width - 1)
                tap_mask = np.zeros_like(component_mask)
                cv2.line(tap_mask, tap_start, tap_end, 255, thickness=(tap_half_width * 2 + 1))
                if not self._allow_candidate_overlap(
                    local_occupied,
                    tap_mask,
                    canvas_shape,
                    [tap_start],
                    allowance_radius=max(2, tap_half_width + 1),
                ):
                    continue
                component_mask = np.maximum(component_mask, tap_mask)
                local_occupied = self._build_guard_mask(component_mask)
                anchors.append(tap_end)

                if rng.random() < 0.45:
                    elbow_mask, elbow_end = self._build_ic_l_stub(
                        canvas_shape,
                        tap_end,
                        rng,
                        x_bounds=(max(1, min(branch_start[0], branch_end[0]) - 24), min(width - 2, max(branch_start[0], branch_end[0]) + 24)),
                        y_bounds=(stripe_min + 1, stripe_max - 1),
                        thickness=max(1, tap_half_width * 2 - 1),
                    )
                    if elbow_mask is not None and self._allow_candidate_overlap(
                        local_occupied,
                        elbow_mask,
                        canvas_shape,
                        [tap_end],
                        allowance_radius=max(2, tap_half_width + 1),
                    ):
                        component_mask = np.maximum(component_mask, elbow_mask)
                        local_occupied = self._build_guard_mask(component_mask)
                        anchors.append(elbow_end)

        for endpoint in (trunk_points[0], trunk_points[-1]):
            if rng.random() < 0.8:
                stub_mask, stub_end = self._build_ic_l_stub(
                    canvas_shape,
                    endpoint,
                    rng,
                    x_bounds=(1, width - 2),
                    y_bounds=(stripe_min + 1, stripe_max - 1),
                    thickness=max(1, trace_half_width * 2 - 1),
                )
                if stub_mask is None or not self._allow_candidate_overlap(
                    local_occupied,
                    stub_mask,
                    canvas_shape,
                    [endpoint],
                    allowance_radius=max(2, trace_half_width + 1),
                ):
                    continue
                component_mask = np.maximum(component_mask, stub_mask)
                local_occupied = self._build_guard_mask(component_mask)
                anchors.append(stub_end)
        return component_mask, anchors
