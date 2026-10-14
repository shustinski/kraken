"""synthetic_topology: build ic cell array."""
from __future__ import annotations
import random
import numpy as np


class BuildIcCellArrayMixin:
    def _build_ic_cell_array_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        profile = self._build_ic_routing_profile(height, width)
        routing_state = self._create_ic_routing_state(height, width, profile)
        row_count = max(4, min(max(4, routing_state['grid_height'] - 1), int(round(max(4, float(self.params.trace_count)) * 0.55))))
        centers = self._resolve_irregular_axis_positions(
            axis_extent=height,
            spacing=max(3, profile['track_step'] * 4),
            requested=row_count,
            rng=rng,
        )
        self._place_ic_cell_blockages(routing_state, centers, rng)
        nets: list[tuple[tuple[int, int], tuple[int, int]]] = []
        group_span = max(2, routing_state['grid_width'] // 9)
        for row_index in range(max(0, len(centers) - 1)):
            top_gy = self._nearest_grid_y(centers[row_index], routing_state['ys'])
            bottom_gy = self._nearest_grid_y(centers[row_index + 1], routing_state['ys'])
            anchor_count = max(2, min(8, int(round(self.params.trace_count / max(1, len(centers) - 1)))))
            routes_per_anchor = max(2, min(8, int(round(self.params.trace_count / max(1, anchor_count)))))
            anchor_positions = np.linspace(1, routing_state['grid_width'] - 2, anchor_count, dtype=int)
            for anchor_x in anchor_positions:
                resolved_anchor_x = int(np.clip(int(anchor_x) + rng.randint(-group_span, group_span), 1, routing_state['grid_width'] - 2))
                for _ in range(routes_per_anchor):
                    start = (int(np.clip(resolved_anchor_x + rng.randint(-group_span, group_span), 0, routing_state['grid_width'] - 1)), top_gy)
                    end = (int(np.clip(resolved_anchor_x + rng.randint(-group_span, group_span), 0, routing_state['grid_width'] - 1)), bottom_gy)
                    if start != end:
                        nets.append((start, end))
                    if rng.random() < 0.55:
                        side_y = top_gy if rng.random() < 0.5 else bottom_gy
                        side_start = (int(np.clip(resolved_anchor_x + rng.randint(-group_span, group_span), 0, routing_state['grid_width'] - 1)), side_y)
                        side_end = (int(np.clip(side_start[0] + rng.randint(-group_span * 2, group_span * 2), 0, routing_state['grid_width'] - 1)), side_y)
                        if side_start != side_end:
                            nets.append((side_start, side_end))
                    if rng.random() < 0.65:
                        jog_y = int(np.clip(((top_gy + bottom_gy) // 2) + rng.randint(-1, 1), 1, routing_state['grid_height'] - 2))
                        jog_start = (int(np.clip(resolved_anchor_x + rng.randint(-group_span, group_span), 0, routing_state['grid_width'] - 1)), top_gy)
                        jog_end = (int(np.clip(jog_start[0] + rng.randint(-group_span * 2, group_span * 2), 0, routing_state['grid_width'] - 1)), bottom_gy)
                        if jog_start != jog_end:
                            nets.append((jog_start, jog_end))
            if row_index % 2 == 0:
                nets.append(((0, top_gy), (routing_state['grid_width'] - 1, top_gy)))
            else:
                nets.append(((0, bottom_gy), (routing_state['grid_width'] - 1, bottom_gy)))
        mask_u8, anchors = self._route_ic_nets(routing_state, nets, rng, family='cell_array')
        if int(np.count_nonzero(mask_u8)) <= 0:
            return self._build_ic_channel_family_mask(height, width, rng)
        return mask_u8, anchors
