"""synthetic_topology: build ic channel row."""
from __future__ import annotations
import random
import cv2
import numpy as np


class BuildIcChannelRowMixin:
    def _build_ic_row_component(
        self,
        canvas_shape: tuple[int, int],
        row_center: int,
        stripe_min: int,
        stripe_max: int,
        rng: random.Random,
        *,
        style: str,
    ) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
        if style == 'cell_array':
            return self._build_ic_cell_row_component(canvas_shape, row_center, stripe_min, stripe_max, rng)
        return self._build_ic_channel_row_component(canvas_shape, row_center, stripe_min, stripe_max, rng)

    def _build_ic_channel_row_component(
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
        if stripe_height < max(12, trace_half_width * 6):
            return None, []
        component_mask = np.zeros(canvas_shape, dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        margin_x = max(10, trace_half_width * 5)
        cluster_count = max(2, min(4, int(round(self.params.trace_count / 3.0))))
        cluster_spans = self._plan_ic_window_slots(
            width,
            cluster_count,
            rng,
            style='channels',
            margin=margin_x,
        )
        if not cluster_spans:
            return None, []
        local_occupied = np.zeros_like(component_mask)
        for cluster_index, (cluster_x1, cluster_x2) in enumerate(cluster_spans):
            span_left = int(np.clip(cluster_x1 - rng.randint(0, trace_half_width + 3), margin_x, width - margin_x - 8))
            span_right = int(np.clip(cluster_x2 + rng.randint(0, trace_half_width + 3), span_left + 12, width - margin_x - 1))
            if span_right - span_left < 18:
                continue
            gap = max(trace_half_width * 4 + 5, int(round(stripe_height * 0.22)))
            top_base = int(np.clip(row_y - (gap // 2), stripe_min + trace_half_width + 1, row_y - trace_half_width - 2))
            bottom_base = int(np.clip(row_y + (gap // 2), row_y + trace_half_width + 2, stripe_max - trace_half_width - 1))
            if bottom_base - top_base < max(8, trace_half_width * 4):
                continue
            top_points = self._build_ic_stepwise_backbone(
                span_left,
                span_right,
                top_base,
                stripe_min + trace_half_width + 1,
                row_y - trace_half_width - 2,
                rng,
                turn_count=max(3, self._segment_range()[1] + 1),
            )
            bottom_points = self._build_ic_stepwise_backbone(
                span_left,
                span_right,
                bottom_base,
                row_y + trace_half_width + 2,
                stripe_max - trace_half_width - 1,
                rng,
                turn_count=max(3, self._segment_range()[1] + 1),
            )
            cluster_mask = np.zeros_like(component_mask)
            self._draw_polyline(cluster_mask, top_points, thickness=(trace_half_width * 2 + 1))
            self._draw_polyline(cluster_mask, bottom_points, thickness=(trace_half_width * 2 + 1))
            bridge_slots = self._plan_ic_window_slots(
                span_right - span_left + 1,
                max(1, min(3, self._segment_range()[1])),
                rng,
                style='channels',
                margin=max(4, trace_half_width * 3),
            )
            for bridge_index, (local_x1, local_x2) in enumerate(bridge_slots):
                left_x = span_left + local_x1
                right_x = span_left + local_x2
                bridge_center_x = int(round((left_x + right_x) / 2.0))
                upper_y = self._resolve_polyline_y_at_x(top_points, bridge_center_x)
                lower_y = self._resolve_polyline_y_at_x(bottom_points, bridge_center_x)
                if lower_y - upper_y < max(8, trace_half_width * 4):
                    continue
                if bridge_index % 2 == 0 and (right_x - left_x) >= 10:
                    band_mask, _band_anchors = self._build_ic_serpentine_rect(
                        canvas_shape,
                        left_x,
                        upper_y + trace_half_width + 1,
                        right_x,
                        lower_y - trace_half_width - 1,
                        max(1, trace_half_width - 1),
                        rng,
                        density_scale=0.78,
                    )
                    if band_mask is not None:
                        cluster_mask = np.maximum(cluster_mask, band_mask)
                else:
                    cv2.line(
                        cluster_mask,
                        (bridge_center_x, upper_y),
                        (bridge_center_x, lower_y),
                        255,
                        thickness=max(1, trace_half_width * 2 - 1),
                    )
            if not self._allow_candidate_overlap(
                local_occupied,
                cluster_mask,
                canvas_shape,
                [],
                allowance_radius=0,
            ):
                continue
            component_mask = np.maximum(component_mask, cluster_mask)
            local_occupied = self._build_guard_mask(component_mask)
            anchors.extend((*top_points, *bottom_points))
            for endpoint in (top_points[0], top_points[-1], bottom_points[0], bottom_points[-1]):
                if rng.random() < 0.8:
                    stub_mask, stub_end = self._build_ic_l_stub(
                        canvas_shape,
                        endpoint,
                        rng,
                        x_bounds=(span_left, span_right),
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
