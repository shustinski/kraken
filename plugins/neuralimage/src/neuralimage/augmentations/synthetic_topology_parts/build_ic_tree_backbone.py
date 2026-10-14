"""synthetic_topology: build ic tree backbone."""
from __future__ import annotations
import random
import cv2
import numpy as np


class BuildIcTreeBackboneMixin:
    def _build_ic_tree_backbone_points(
        self,
        width: int,
        row_y: int,
        stripe_min: int,
        stripe_max: int,
        rng: random.Random,
    ) -> list[tuple[int, int]]:
        stripe_height = max(1, stripe_max - stripe_min)
        usable_top = max(stripe_min + 2, row_y - max(6, int(round(stripe_height * 0.42))))
        usable_bottom = min(stripe_max - 2, row_y + max(6, int(round(stripe_height * 0.42))))
        x_margin = max(10, min(width // 6, (int(self.params.margin) * 2) + self._width_range()[1] + 3))
        left_x = rng.randint(x_margin // 2, max(x_margin // 2 + 1, x_margin))
        right_x = rng.randint(max(left_x + 16, width - x_margin), max(left_x + 16, width - (x_margin // 2) - 1))
        segment_count = max(4, min(8, self._segment_range()[1] + 2))
        breakpoints = np.linspace(left_x, right_x, segment_count + 1)
        points: list[tuple[int, int]] = []
        current_y = int(np.clip(
            row_y + rng.randint(-max(3, stripe_height // 10), max(3, stripe_height // 10)),
            usable_top,
            usable_bottom,
        ))
        points.append((left_x, current_y))
        last_direction = 1 if rng.random() < 0.5 else -1
        max_delta = max(4, int(round((usable_bottom - usable_top) * 0.36)))
        min_delta = 2
        for breakpoint_index, breakpoint in enumerate(breakpoints[1:-1], start=1):
            x = int(round(float(breakpoint)))
            preferred_direction = -last_direction if (breakpoint_index % 2 == 1) else last_direction
            delta = rng.randint(max(2, max_delta // 3), max_delta)
            candidate_y = current_y + (preferred_direction * delta)
            if candidate_y < usable_top or candidate_y > usable_bottom:
                preferred_direction *= -1
                candidate_y = current_y + (preferred_direction * delta)
            next_y = int(np.clip(candidate_y, usable_top, usable_bottom))
            if abs(next_y - current_y) < min_delta:
                next_y = int(np.clip(
                    current_y + (preferred_direction * min_delta),
                    usable_top,
                    usable_bottom,
                ))
            points.append((x, current_y))
            if next_y != current_y:
                points.append((x, next_y))
                last_direction = 1 if next_y > current_y else -1
            current_y = next_y
        points.append((right_x, current_y))
        return points

    def _build_ic_stepwise_backbone(
        self,
        left_x: int,
        right_x: int,
        base_y: int,
        y_min: int,
        y_max: int,
        rng: random.Random,
        *,
        turn_count: int,
    ) -> list[tuple[int, int]]:
        constrained_min = min(y_min, y_max)
        constrained_max = max(y_min, y_max)
        current_y = int(np.clip(base_y, constrained_min, constrained_max))
        resolved_left = min(left_x, right_x)
        resolved_right = max(left_x, right_x)
        if resolved_right - resolved_left < 8:
            resolved_right = resolved_left + 8
        points: list[tuple[int, int]] = [(resolved_left, current_y)]
        breakpoints = np.linspace(resolved_left, resolved_right, max(3, int(turn_count)))
        amplitude = max(3, int(round((constrained_max - constrained_min) * 0.34)))
        last_direction = 1 if rng.random() < 0.5 else -1
        for index, breakpoint in enumerate(breakpoints[1:-1], start=1):
            x = int(round(float(breakpoint)))
            target_y = current_y + (last_direction * rng.randint(max(2, amplitude // 3), amplitude))
            if target_y < constrained_min or target_y > constrained_max:
                last_direction *= -1
                target_y = current_y + (last_direction * rng.randint(max(2, amplitude // 3), amplitude))
            target_y = int(np.clip(target_y, constrained_min, constrained_max))
            if abs(target_y - current_y) < 2 and (index % 2 == 1):
                target_y = int(np.clip(current_y + (last_direction * 2), constrained_min, constrained_max))
            points.append((x, current_y))
            if target_y != current_y:
                points.append((x, target_y))
                last_direction = 1 if target_y > current_y else -1
            current_y = target_y
        points.append((resolved_right, current_y))
        return points

    def _build_ic_l_stub(
        self,
        canvas_shape: tuple[int, int],
        start_xy: tuple[int, int],
        rng: random.Random,
        *,
        x_bounds: tuple[int, int],
        y_bounds: tuple[int, int],
        thickness: int,
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        height, width = canvas_shape
        x_min = int(np.clip(min(x_bounds), 0, width - 1))
        x_max = int(np.clip(max(x_bounds), 0, width - 1))
        y_min = int(np.clip(min(y_bounds), 0, height - 1))
        y_max = int(np.clip(max(y_bounds), 0, height - 1))
        horizontal_length = rng.randint(6, max(7, min(24, max(6, (x_max - x_min) // 3))))
        vertical_length = rng.randint(6, max(7, min(24, max(6, (y_max - y_min) // 3))))
        horizontal_direction = -1 if rng.random() < 0.5 else 1
        vertical_direction = -1 if rng.random() < 0.5 else 1
        elbow = (
            int(np.clip(start_xy[0] + (horizontal_direction * horizontal_length), x_min, x_max)),
            int(start_xy[1]),
        )
        end_point = (
            int(elbow[0]),
            int(np.clip(elbow[1] + (vertical_direction * vertical_length), y_min, y_max)),
        )
        if elbow == start_xy or end_point == elbow:
            return None, start_xy
        stub_mask = np.zeros(canvas_shape, dtype=np.uint8)
        cv2.line(stub_mask, start_xy, elbow, 255, thickness=max(1, int(thickness)))
        cv2.line(stub_mask, elbow, end_point, 255, thickness=max(1, int(thickness)))
        return stub_mask, end_point

    def _build_ic_tree_branch_polyline(
        self,
        start_xy: tuple[int, int],
        width: int,
        stripe_min: int,
        stripe_max: int,
        rng: random.Random,
        *,
        direction: int,
        max_length: int,
    ) -> list[tuple[int, int]]:
        vertical_target_y = int(np.clip(
            start_xy[1] + (direction * max(8, int(round(max_length * rng.uniform(0.42, 0.78))))),
            stripe_min + 1,
            stripe_max - 1,
        ))
        if vertical_target_y == start_xy[1]:
            return [start_xy]
        elbow_x = int(np.clip(
            start_xy[0] + rng.randint(-max(8, width // 10), max(8, width // 10)),
            0,
            width - 1,
        ))
        end_x = int(np.clip(
            elbow_x + rng.randint(-max(10, width // 8), max(10, width // 8)),
            0,
            width - 1,
        ))
        points = [start_xy, (start_xy[0], vertical_target_y)]
        if elbow_x != start_xy[0]:
            points.append((elbow_x, vertical_target_y))
        if end_x != points[-1][0]:
            points.append((end_x, vertical_target_y))
        return points
