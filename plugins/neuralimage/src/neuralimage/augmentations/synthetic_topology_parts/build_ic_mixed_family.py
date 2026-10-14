"""synthetic_topology: build ic mixed family."""
from __future__ import annotations
import math
import random
from dataclasses import replace
import numpy as np


class BuildIcMixedFamilyMixin:
    def _build_ic_mixed_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        combined_anchors: list[tuple[int, int]] = []
        family_specs = (
            ('ic_channels', max(4, int(round(self.params.trace_count * 0.36)))),
            ('ic_cell_array', max(4, int(round(self.params.trace_count * 0.34)))),
            ('ic_tree', max(3, int(round(self.params.trace_count * 0.30)))),
        )
        for family_name, family_trace_count in family_specs:
            placed = False
            for _attempt_index in range(8):
                child_params = replace(
                    self.params,
                    topology_domain='ic',
                    topology_family=family_name,
                    trace_count=max(1, family_trace_count),
                )
                child_generator = SyntheticTopologyGenerator(child_params)
                family_mask, family_anchors = child_generator._build_family_mask(
                    child_generator._resolve_family(rng),
                    height,
                    width,
                    rng,
                )
                if int(np.count_nonzero(family_mask)) <= 0:
                    continue
                if self._mask_overlaps_guard(
                    family_mask,
                    combined_mask,
                    radius=max(1, int(self.params.trace_clearance)),
                ):
                    continue
                combined_mask = np.maximum(combined_mask, family_mask)
                combined_anchors.extend(
                    anchor
                    for anchor in family_anchors
                    if 0 <= int(anchor[0]) < width and 0 <= int(anchor[1]) < height and family_mask[int(anchor[1]), int(anchor[0])] > 0
                )
                placed = True
                break
            if not placed:
                continue
        if int(np.count_nonzero(combined_mask)) <= 0:
            return self._build_ic_channel_family_mask(height, width, rng)
        return combined_mask, combined_anchors

    def _build_ic_channel_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        profile = self._build_ic_routing_profile(height, width)
        routing_state = self._create_ic_routing_state(height, width, profile)
        channel_count = max(3, min(max(3, routing_state['grid_height'] - 2), int(round(max(3, float(self.params.trace_count)) * 0.65))))
        channel_centers = self._resolve_irregular_axis_positions(
            axis_extent=height,
            spacing=max(2, profile['track_step'] * 3),
            requested=channel_count,
            rng=rng,
        )
        self._place_ic_channel_blockages(routing_state, channel_centers, rng)
        nets: list[tuple[tuple[int, int], tuple[int, int]]] = []
        spanning_net_budget = max(1, int(round(self.params.trace_count * 0.12)))
        for center_y in channel_centers:
            cluster_count = max(1, min(5, int(round(self.params.trace_count / 24.0))))
            top_track = self._nearest_grid_y(center_y - profile['track_step'], routing_state['ys'])
            bottom_track = self._nearest_grid_y(center_y + profile['track_step'], routing_state['ys'])
            routes_per_cluster = max(3, int(math.ceil(max(6.0, float(self.params.trace_count) * 0.55) / max(1, len(channel_centers) * cluster_count))))
            for cluster_index in range(cluster_count):
                cluster_span = self._randint_safe(rng, 2, max(4, routing_state['grid_width'] // max(4, cluster_count)))
                cluster_anchor = int(round(((cluster_index + 1) / float(cluster_count + 1)) * (routing_state['grid_width'] - 1)))
                cluster_left = max(0, cluster_anchor - cluster_span)
                cluster_right = min(routing_state['grid_width'] - 1, cluster_anchor + cluster_span)
                for _ in range(routes_per_cluster):
                    start_x = rng.randint(cluster_left, cluster_right)
                    end_x = rng.randint(cluster_left, cluster_right)
                    start = (start_x, top_track if rng.random() < 0.5 else bottom_track)
                    end = (end_x, bottom_track if start[1] == top_track else top_track)
                    if start != end:
                        nets.append((start, end))
                    if rng.random() < 0.72:
                        mid_track = int(np.clip((top_track + bottom_track) // 2 + rng.randint(-1, 1), 1, routing_state['grid_height'] - 2))
                        branch_start = (rng.randint(cluster_left, cluster_right), mid_track)
                        branch_end = (rng.randint(cluster_left, cluster_right), top_track if rng.random() < 0.5 else bottom_track)
                        if branch_start != branch_end:
                            nets.append((branch_start, branch_end))
            for _ in range(max(1, spanning_net_budget // max(1, len(channel_centers)))):
                start = (0, top_track if rng.random() < 0.5 else bottom_track)
                end = (routing_state['grid_width'] - 1, bottom_track if start[1] == top_track else top_track)
                nets.append((start, end))
        mask_u8, anchors = self._route_ic_nets(routing_state, nets, rng, family='channels')
        if int(np.count_nonzero(mask_u8)) <= 0:
            return self._build_ic_tree_family_mask(height, width, rng)
        return mask_u8, anchors
