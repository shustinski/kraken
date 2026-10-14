"""synthetic_topology: build ic cell row."""
from __future__ import annotations
import random
import cv2
import numpy as np


class BuildIcCellRowMixin:
    def _build_ic_cell_row_component(
        self,
        canvas_shape: tuple[int, int],
        row_center: int,
        stripe_min: int,
        stripe_max: int,
        rng: random.Random,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        _height, width = canvas_shape
        trace_half_width = rng.randint(*self._width_range())
        row_y = int(np.clip(row_center, stripe_min + trace_half_width + 2, stripe_max - trace_half_width - 2))
        stripe_height = max(1, stripe_max - stripe_min)
        if stripe_height < max(14, trace_half_width * 6):
            return None, []
        component_mask = np.zeros(canvas_shape, dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        margin_x = max(12, trace_half_width * 6)
        group_count = max(2, min(4, int(round(self.params.trace_count / 3.5))))
        group_spans = self._plan_ic_window_slots(
            width,
            group_count,
            rng,
            style='channels',
            margin=margin_x,
        )
        if not group_spans:
            return None, []
        local_occupied = np.zeros_like(component_mask)
        for group_index, (group_x1, group_x2) in enumerate(group_spans):
            group_left = int(np.clip(group_x1 - rng.randint(0, trace_half_width + 3), margin_x, width - margin_x - 10))
            group_right = int(np.clip(group_x2 + rng.randint(0, trace_half_width + 3), group_left + 18, width - margin_x - 1))
            if group_right - group_left < 20:
                continue
            group_top = max(stripe_min + 1, row_y - max(7, int(round(stripe_height * 0.30))))
            group_bottom = min(stripe_max - 1, row_y + max(7, int(round(stripe_height * 0.30))))
            rail_offset = max(trace_half_width + 2, int(round((group_bottom - group_top) * 0.22)))
            upper_rail_y = int(np.clip(row_y - rail_offset, group_top, row_y - 2))
            lower_rail_y = int(np.clip(row_y + rail_offset, row_y + 2, group_bottom))
            group_mask = np.zeros_like(component_mask)
            cv2.line(group_mask, (group_left, upper_rail_y), (group_right, upper_rail_y), 255, thickness=(trace_half_width * 2 + 1))
            cv2.line(group_mask, (group_left, lower_rail_y), (group_right, lower_rail_y), 255, thickness=(trace_half_width * 2 + 1))
            anchors.extend(((group_left, upper_rail_y), (group_right, upper_rail_y), (group_left, lower_rail_y), (group_right, lower_rail_y)))
            cell_count = max(2, min(6, int((group_right - group_left) / max(16, trace_half_width * 8))))
            cell_pitch = (group_right - group_left) / float(max(1, cell_count))
            previous_right_anchor: tuple[int, int] | None = None
            for cell_index in range(cell_count):
                center = group_left + ((cell_index + 0.5) * cell_pitch)
                cell_width = max(14, int(round(cell_pitch * rng.uniform(0.64, 0.88))))
                cell_x1 = int(round(center - (cell_width / 2.0)))
                cell_x2 = int(round(center + (cell_width / 2.0)))
                cell_x1 = max(group_left + 1, cell_x1)
                cell_x2 = min(group_right - 1, cell_x2)
                if cell_x2 - cell_x1 < 12:
                    continue
                motif_mask, motif_anchors = self._build_ic_cell_motif(
                    canvas_shape,
                    cell_x1,
                    group_top,
                    cell_x2,
                    group_bottom,
                    rng,
                )
                if motif_mask is None or len(motif_anchors) < 2:
                    continue
                left_anchor, right_anchor = motif_anchors[0], motif_anchors[1]
                connector_mask = np.zeros_like(component_mask)
                strap_x = int(round((left_anchor[0] + right_anchor[0]) / 2.0))
                target_rail_y = upper_rail_y if (cell_index % 2 == 0) else lower_rail_y
                cv2.line(
                    connector_mask,
                    (strap_x, row_y),
                    (strap_x, target_rail_y),
                    255,
                    thickness=max(1, trace_half_width * 2 - 1),
                )
                if previous_right_anchor is not None:
                    bridge_y = previous_right_anchor[1]
                    cv2.line(
                        connector_mask,
                        previous_right_anchor,
                        (left_anchor[0], bridge_y),
                        255,
                        thickness=max(1, trace_half_width * 2 - 1),
                    )
                    if bridge_y != left_anchor[1]:
                        cv2.line(
                            connector_mask,
                            (left_anchor[0], bridge_y),
                            left_anchor,
                            255,
                            thickness=max(1, trace_half_width * 2 - 1),
                        )
                candidate_mask = np.maximum(motif_mask, connector_mask)
                allowed_points = [left_anchor, right_anchor, (strap_x, row_y), (strap_x, target_rail_y)]
                if previous_right_anchor is not None:
                    allowed_points.append(previous_right_anchor)
                if not self._allow_candidate_overlap(
                    local_occupied,
                    candidate_mask,
                    canvas_shape,
                    allowed_points,
                    allowance_radius=max(2, trace_half_width + 1),
                ):
                    continue
                group_mask = np.maximum(group_mask, candidate_mask)
                previous_right_anchor = right_anchor
                anchors.extend((*motif_anchors, (strap_x, target_rail_y)))
            if not self._allow_candidate_overlap(
                local_occupied,
                group_mask,
                canvas_shape,
                [],
                allowance_radius=0,
            ):
                continue
            component_mask = np.maximum(component_mask, group_mask)
            local_occupied = self._build_guard_mask(component_mask)
            for endpoint in ((group_left, upper_rail_y), (group_right, upper_rail_y), (group_left, lower_rail_y), (group_right, lower_rail_y)):
                if rng.random() < 0.7:
                    stub_mask, stub_end = self._build_ic_l_stub(
                        canvas_shape,
                        endpoint,
                        rng,
                        x_bounds=(group_left, group_right),
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
        return (None, []) if int(np.count_nonzero(component_mask)) <= 0 else (component_mask, anchors)
