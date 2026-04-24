from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticTopologyParameters:
    trace_count: int = 5
    segment_count: int = 4
    trace_half_width: int = 2
    segment_count_range: tuple[int, int] | None = None
    trace_half_width_range: tuple[int, int] | None = None
    margin: int = 12
    background_noise_sigma: float = 0.02
    trace_noise_sigma: float = 0.01
    trace_clearance: int = 2
    topology_domain: str = 'pcb'
    topology_family: str = 'independent'
    via_count_range: tuple[int, int] | None = None


class SyntheticTopologyGenerator:
    """Generate a consistent synthetic topology pair: grayscale image + binary mask."""

    _MIXED_FAMILIES: tuple[tuple[str, float], ...] = (
        ('independent', 0.28),
        ('parallel', 0.28),
        ('tree', 0.22),
        ('cell_array', 0.22),
    )
    _PCB_FAMILIES: dict[str, str] = {
        'pcb_parallel': 'parallel',
        'pcb_independent': 'independent',
        'pcb_fanout': 'tree',
    }
    _IC_FAMILIES: dict[str, str] = {
        'ic_cell_array': 'cell_array',
        'ic_channels': 'parallel',
        'ic_tree': 'tree',
    }
    _PCB_MIXED_FAMILIES: tuple[tuple[str, float], ...] = (
        ('independent', 0.36),
        ('parallel', 0.34),
        ('tree', 0.30),
    )
    _IC_MIXED_FAMILIES: tuple[tuple[str, float], ...] = (
        ('cell_array', 0.42),
        ('parallel', 0.35),
        ('tree', 0.23),
    )

    def __init__(self, params: SyntheticTopologyParameters | None = None) -> None:
        self.params = params or SyntheticTopologyParameters()

    def generate(
        self,
        *,
        size_hw: tuple[int, int],
        channels: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height = max(32, int(size_hw[0]))
        width = max(32, int(size_hw[1]))
        rng = random.Random(int(seed))
        np_rng = np.random.default_rng(int(seed))

        family = self._resolve_family(rng)
        mask_u8, anchors = self._build_family_mask(family, height, width, rng)
        mask_u8 = self._decorate_terminals(mask_u8, anchors, family, rng)

        if not np.any(mask_u8):
            mask_u8 = self._build_fallback_mask(height, width)

        image_array = self._render_image(mask_u8, np_rng)
        image_chw = self._to_channel_first(image_array, channels)
        label_chw = (mask_u8.astype(np.float32) / 255.0)[None, :, :]
        return image_chw.astype(np.float32, copy=False), label_chw.astype(np.float32, copy=False)

    def _resolve_family(self, rng: random.Random) -> str:
        domain = self._domain()
        family = str(getattr(self.params, 'topology_family', 'independent') or 'independent').strip().lower()
        if domain == 'ic' and family in ('ic_mixed', 'mixed'):
            return 'ic_mixed'
        if family in ('pcb_mixed', 'mixed'):
            mixed_population = self._PCB_MIXED_FAMILIES if domain != 'ic' else self._IC_MIXED_FAMILIES
            population = [item[0] for item in mixed_population]
            weights = [item[1] for item in mixed_population]
            return str(rng.choices(population, weights=weights, k=1)[0])
        if family in self._PCB_FAMILIES:
            return self._PCB_FAMILIES[family]
        if family in self._IC_FAMILIES:
            return self._IC_FAMILIES[family]
        if family:
            return family
        population = [item[0] for item in self._MIXED_FAMILIES]
        weights = [item[1] for item in self._MIXED_FAMILIES]
        return str(rng.choices(population, weights=weights, k=1)[0])

    def _build_family_mask(
        self,
        family: str,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        domain = self._domain()
        if domain == 'ic':
            if family == 'ic_mixed':
                return self._build_ic_mixed_family_mask(height, width, rng)
            if family == 'cell_array':
                return self._build_ic_cell_array_family_mask(height, width, rng)
            if family == 'tree':
                return self._build_ic_tree_family_mask(height, width, rng)
            return self._build_ic_channel_family_mask(height, width, rng)
        if family == 'parallel':
            return self._build_pcb_parallel_family_mask(height, width, rng)
        if family == 'tree':
            return self._build_pcb_tree_family_mask(height, width, rng)
        if family == 'cell_array':
            return self._build_pcb_cell_array_family_mask(height, width, rng)
        return self._build_pcb_independent_family_mask(height, width, rng)

    def _domain(self) -> str:
        domain = str(getattr(self.params, 'topology_domain', 'pcb') or 'pcb').strip().lower()
        return 'ic' if domain == 'ic' else 'pcb'

    def _build_pcb_parallel_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        orientation = str(rng.choices(
            ('horizontal', 'vertical', 'diag_pos', 'diag_neg'),
            weights=(0.32, 0.18, 0.25, 0.25),
            k=1,
        )[0])
        requested = max(2, int(self.params.trace_count))
        max_width = self._width_range()[1]
        spacing = max(8, (max_width * 3) + max(2, int(self.params.trace_clearance)) + 3)
        for offset in self._resolve_parallel_offsets(height, width, orientation, spacing, requested):
            trace_half_width = rng.randint(*self._width_range())
            segment = self._full_span_segment(height, width, orientation, offset)
            if segment is None:
                continue
            candidate_mask = np.zeros_like(mask_u8)
            cv2.line(candidate_mask, segment[0], segment[1], 255, thickness=(trace_half_width * 2 + 1))
            if np.any((candidate_mask > 0) & (self._build_guard_mask(mask_u8) > 0)):
                continue
            mask_u8 = np.maximum(mask_u8, candidate_mask)
            anchors.extend((segment[0], segment[1], self._midpoint(segment[0], segment[1])))
        if int(np.count_nonzero(mask_u8)) <= 0:
            return self._build_pcb_independent_family_mask(height, width, rng)
        return mask_u8, anchors

    def _build_pcb_independent_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        requested_trace_count = max(1, int(self.params.trace_count))
        for _trace_index in range(requested_trace_count):
            candidate_mask, candidate_anchors = self._build_pcb_route_component(
                height,
                width,
                rng,
                occupied_guard_mask=self._build_guard_mask(mask_u8),
            )
            if candidate_mask is None:
                continue
            mask_u8 = np.maximum(mask_u8, candidate_mask)
            anchors.extend(candidate_anchors)
        return mask_u8, anchors

    def _build_pcb_tree_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        trace_half_width = rng.randint(*self._width_range())
        horizontal = bool(rng.getrandbits(1))
        margin = self._effective_margin(height, width)
        if horizontal:
            center_y = rng.randint(margin, max(margin + 1, height - margin - 1))
            trunk_start = (0, center_y)
            trunk_end = (width - 1, center_y)
        else:
            center_x = rng.randint(margin, max(margin + 1, width - margin - 1))
            trunk_start = (center_x, 0)
            trunk_end = (center_x, height - 1)
        cv2.line(mask_u8, trunk_start, trunk_end, 255, thickness=(trace_half_width * 2 + 1))
        anchors.extend((trunk_start, trunk_end))

        branch_target = max(2, min(18, int(self.params.trace_count)))
        for branch_fraction in self._resolve_branch_fractions(branch_target):
            branch_half_width = rng.randint(*self._width_range())
            branch_start = self._point_on_segment(trunk_start, trunk_end, branch_fraction)
            if horizontal:
                branch_end = self._build_axis_branch_endpoint(
                    branch_start,
                    length=max(10, int(round(height * rng.uniform(0.18, 0.44)))),
                    direction=-1 if rng.random() < 0.5 else 1,
                    axis='y',
                    width=width,
                    height=height,
                )
            else:
                branch_end = self._build_axis_branch_endpoint(
                    branch_start,
                    length=max(10, int(round(width * rng.uniform(0.18, 0.44)))),
                    direction=-1 if rng.random() < 0.5 else 1,
                    axis='x',
                    width=width,
                    height=height,
                )
            cv2.line(mask_u8, branch_start, branch_end, 255, thickness=(branch_half_width * 2 + 1))
            if rng.random() < 0.6:
                diagonal_end = self._build_pcb_diagonal_branch_endpoint(
                    branch_end,
                    width=width,
                    height=height,
                    rng=rng,
                )
                cv2.line(mask_u8, branch_end, diagonal_end, 255, thickness=(branch_half_width * 2 + 1))
                anchors.append(diagonal_end)
            anchors.append(branch_end)
        return mask_u8, anchors

    def _build_pcb_cell_array_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        anchors: list[tuple[int, int]] = []
        margin = self._effective_margin(height, width)
        motif_count = max(2, int(self.params.trace_count))
        cols = max(2, int(round(np.sqrt(motif_count))))
        rows = max(1, int(np.ceil(float(motif_count) / float(cols))))
        cell_w = max(22, (width - (margin * 2)) // max(1, cols))
        cell_h = max(22, (height - (margin * 2)) // max(1, rows))
        motif_index = 0
        for row in range(rows):
            for col in range(cols):
                if motif_index >= motif_count:
                    break
                x1 = margin + (col * cell_w)
                y1 = margin + (row * cell_h)
                x2 = min(width - margin, x1 + cell_w)
                y2 = min(height - margin, y1 + cell_h)
                candidate_mask, candidate_anchors = self._build_pcb_cell_motif(mask_u8.shape, x1, y1, x2, y2, rng)
                if candidate_mask is None:
                    motif_index += 1
                    continue
                if np.any((candidate_mask > 0) & (self._build_guard_mask(mask_u8) > 0)):
                    motif_index += 1
                    continue
                mask_u8 = np.maximum(mask_u8, candidate_mask)
                anchors.extend(candidate_anchors)
                motif_index += 1
        if int(np.count_nonzero(mask_u8)) <= 0:
            return self._build_pcb_parallel_family_mask(height, width, rng)
        return mask_u8, anchors

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

    def _build_ic_tree_family_mask(
        self,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        profile = self._build_ic_routing_profile(height, width)
        routing_state = self._create_ic_routing_state(height, width, profile)
        tree_count = max(2, min(max(2, routing_state['grid_height'] - 2), int(round(self.params.trace_count / 10.0)) + 1))
        centers = self._resolve_irregular_axis_positions(
            axis_extent=height,
            spacing=max(3, profile['track_step'] * 4),
            requested=tree_count,
            rng=rng,
        )
        anchors: list[tuple[int, int]] = []
        for tree_index, center_y in enumerate(centers):
            use_full_span = (tree_index % 2 == 0) or (rng.random() < 0.35)
            if use_full_span:
                left_gx = 0
                right_gx = routing_state['grid_width'] - 1
            else:
                left_gx = self._randint_safe(rng, 1, max(2, routing_state['grid_width'] // 4))
                right_gx = self._randint_safe(
                    rng,
                    min(routing_state['grid_width'] - 2, left_gx + 3),
                    max(left_gx + 3, routing_state['grid_width'] - 2),
                )
            trunk_gy = self._nearest_grid_y(center_y, routing_state['ys'])
            trunk_path = self._pattern_route((left_gx, trunk_gy), (right_gx, trunk_gy), routing_state['blocked'], family='tree', rng=rng)
            if trunk_path is None:
                trunk_path = self._maze_route((left_gx, trunk_gy), (right_gx, trunk_gy), routing_state['blocked'], family='tree', rng=rng)
            if not trunk_path:
                continue
            trunk_half_width = self._sample_ic_route_half_width(rng)
            trunk_mask = self._build_grid_path_mask(
                mask_shape=np.asarray(routing_state['mask'], dtype=np.uint8).shape,
                path=trunk_path,
                routing_state=routing_state,
                thickness=(trunk_half_width * 2 + 1),
            )
            if self._mask_overlaps_guard(
                trunk_mask,
                np.asarray(routing_state['mask'], dtype=np.uint8),
                radius=max(1, int(self.params.trace_clearance)),
            ):
                continue
            self._occupy_grid_path(
                routing_state['blocked'],
                trunk_path,
                padding=max(1, self._grid_padding_for_half_width(routing_state, trunk_half_width)),
            )
            np.asarray(routing_state['mask'], dtype=np.uint8)[:] = np.maximum(np.asarray(routing_state['mask'], dtype=np.uint8), trunk_mask)
            anchors.extend(self._grid_path_to_pixels(trunk_path, routing_state))
            branch_count = max(4, min(20, int(round(self.params.trace_count * 0.45))))
            for fraction in self._resolve_branch_fractions(branch_count):
                branch_start = trunk_path[min(len(trunk_path) - 1, int(round((len(trunk_path) - 1) * fraction)))]
                direction = -1 if rng.random() < 0.5 else 1
                sink_gy = int(np.clip(
                    branch_start[1] + (
                        direction * self._randint_safe(rng, max(2, routing_state['grid_height'] // 8), max(4, routing_state['grid_height'] // 3))
                    ),
                    1,
                    routing_state['grid_height'] - 2,
                ))
                sink_gx = int(np.clip(branch_start[0] + rng.randint(-max(2, routing_state['grid_width'] // 10), max(2, routing_state['grid_width'] // 10)), 1, routing_state['grid_width'] - 2))
                branch_path = self._route_grid_path_to_tree(
                    sink=(sink_gx, sink_gy),
                    tree_nodes=set(trunk_path),
                    blocked=routing_state['blocked'],
                    family='tree',
                    rng=rng,
                )
                if not branch_path or len(branch_path) < 2:
                    continue
                branch_half_width = self._sample_ic_route_half_width(rng)
                branch_mask = self._build_grid_path_mask(
                    mask_shape=np.asarray(routing_state['mask'], dtype=np.uint8).shape,
                    path=branch_path,
                    routing_state=routing_state,
                    thickness=(branch_half_width * 2 + 1),
                )
                branch_attach_point = self._grid_path_to_pixels([branch_path[-1]], routing_state)[0]
                if self._mask_overlaps_guard(
                    branch_mask,
                    np.asarray(routing_state['mask'], dtype=np.uint8),
                    allowed_points=[branch_attach_point],
                    allowance_radius=max(branch_half_width + 1, self._width_range()[0]),
                    radius=max(1, int(self.params.trace_clearance)),
                ):
                    continue
                self._occupy_grid_path(
                    routing_state['blocked'],
                    branch_path,
                    padding=max(1, self._grid_padding_for_half_width(routing_state, branch_half_width)),
                )
                np.asarray(routing_state['mask'], dtype=np.uint8)[:] = np.maximum(np.asarray(routing_state['mask'], dtype=np.uint8), branch_mask)
                anchors.extend(self._grid_path_to_pixels(branch_path, routing_state))
                for tap_fraction in np.linspace(0.25, 0.85, 2 + int(rng.random() < 0.5)):
                    tap_origin = branch_path[min(len(branch_path) - 1, int(round((len(branch_path) - 1) * float(tap_fraction))))]
                    tap_side = -1 if rng.random() < 0.5 else 1
                    tap_sink = (
                        int(np.clip(tap_origin[0] + (tap_side * rng.randint(2, max(3, routing_state['grid_width'] // 10))), 1, routing_state['grid_width'] - 2)),
                        tap_origin[1],
                    )
                    tap_path = self._pattern_route(tap_origin, tap_sink, routing_state['blocked'], family='tree', rng=rng)
                    if tap_path is None:
                        tap_path = self._maze_route(tap_origin, tap_sink, routing_state['blocked'], family='tree', rng=rng)
                    if not tap_path or len(tap_path) < 2:
                        continue
                    tap_half_width = max(1, self._sample_ic_route_half_width(rng) - 1)
                    tap_mask = self._build_grid_path_mask(
                        mask_shape=np.asarray(routing_state['mask'], dtype=np.uint8).shape,
                        path=tap_path,
                        routing_state=routing_state,
                        thickness=max(1, tap_half_width * 2 + 1),
                    )
                    tap_attach_point = self._grid_path_to_pixels([tap_path[0]], routing_state)[0]
                    if self._mask_overlaps_guard(
                        tap_mask,
                        np.asarray(routing_state['mask'], dtype=np.uint8),
                        allowed_points=[tap_attach_point],
                        allowance_radius=max(tap_half_width + 1, self._width_range()[0]),
                        radius=max(1, int(self.params.trace_clearance)),
                    ):
                        continue
                    self._occupy_grid_path(
                        routing_state['blocked'],
                        tap_path,
                        padding=self._grid_padding_for_half_width(routing_state, tap_half_width),
                    )
                    np.asarray(routing_state['mask'], dtype=np.uint8)[:] = np.maximum(np.asarray(routing_state['mask'], dtype=np.uint8), tap_mask)
                    anchors.extend(self._grid_path_to_pixels(tap_path, routing_state))
        return routing_state['mask'], anchors

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

    def _pattern_route(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        blocked: np.ndarray,
        *,
        family: str,
        rng: random.Random,
    ) -> list[tuple[int, int]] | None:
        sx, sy = start
        ex, ey = end
        height, width = blocked.shape
        candidates: list[list[tuple[int, int]]] = []
        if sy == ey or sx == ex:
            candidates.append([start, end])
        else:
            candidates.append([start, (ex, sy), end])
            candidates.append([start, (sx, ey), end])
            detour_x = int(round((sx + ex) / 2.0))
            detour_y = int(round((sy + ey) / 2.0))
            candidates.append([start, (detour_x, sy), (detour_x, ey), end])
            candidates.append([start, (sx, detour_y), (ex, detour_y), end])
            if family == 'tree':
                mid_x = int(round((sx + ex) / 2.0))
                mid_y = int(round((sy + ey) / 2.0))
                candidates.append([start, (mid_x, sy), (mid_x, ey), end])
                candidates.append([start, (sx, mid_y), (ex, mid_y), end])
            elif family in ('channels', 'cell_array'):
                offset = 2 if abs(sx - ex) > 4 else 1
                left_detour_x = int(np.clip(sx + offset, 0, width - 1))
                right_detour_x = int(np.clip(ex - offset, 0, width - 1))
                candidates.append([start, (left_detour_x, sy), (left_detour_x, ey), end])
                candidates.append([start, (right_detour_x, sy), (right_detour_x, ey), end])
        rng.shuffle(candidates)
        for points in candidates:
            grid_path = self._expand_grid_polyline(points)
            if grid_path and self._grid_path_is_free(grid_path, blocked, allowed={start, end}):
                return grid_path
        return None

    def _maze_route(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        blocked: np.ndarray,
        *,
        family: str,
        rng: random.Random,
    ) -> list[tuple[int, int]] | None:
        height, width = blocked.shape
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        frontier = [start]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        order_bias = 1 if family == 'channels' else -1
        while frontier:
            current = frontier.pop(0)
            if current == end:
                break
            neighbors = self._grid_neighbors(current, width, height)
            neighbors.sort(key=lambda point: (abs(point[0] - end[0]) + abs(point[1] - end[1])) + (order_bias * abs(point[1] - current[1])))
            for neighbor in neighbors:
                if neighbor in came_from:
                    continue
                if blocked[neighbor[1], neighbor[0]] > 0 and neighbor != end:
                    continue
                came_from[neighbor] = current
                frontier.append(neighbor)
        if end not in came_from:
            return None
        path = [end]
        cursor = end
        while came_from[cursor] is not None:
            cursor = came_from[cursor]
            path.append(cursor)
        path.reverse()
        return self._compress_grid_path(path)

    @staticmethod
    def _grid_neighbors(point: tuple[int, int], width: int, height: int) -> list[tuple[int, int]]:
        x, y = point
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [(nx, ny) for nx, ny in candidates if 0 <= nx < width and 0 <= ny < height]

    def _route_grid_path_to_tree(
        self,
        *,
        sink: tuple[int, int],
        tree_nodes: set[tuple[int, int]],
        blocked: np.ndarray,
        family: str,
        rng: random.Random,
    ) -> list[tuple[int, int]] | None:
        best_path: list[tuple[int, int]] | None = None
        best_cost: int | None = None
        for tree_node in tree_nodes:
            path = self._pattern_route(sink, tree_node, blocked, family=family, rng=rng)
            if path is None:
                path = self._maze_route(sink, tree_node, blocked, family=family, rng=rng)
            if not path:
                continue
            cost = len(path)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path

    @staticmethod
    def _expand_grid_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(points) < 2:
            return []
        expanded: list[tuple[int, int]] = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            sx, sy = start
            ex, ey = end
            if sx != ex and sy != ey:
                return []
            step_x = 0 if sx == ex else (1 if ex > sx else -1)
            step_y = 0 if sy == ey else (1 if ey > sy else -1)
            cx, cy = sx, sy
            while (cx, cy) != (ex, ey):
                cx += step_x
                cy += step_y
                expanded.append((cx, cy))
        return expanded

    @staticmethod
    def _compress_grid_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(path) <= 2:
            return path[:]
        compressed = [path[0]]
        last_dx = path[1][0] - path[0][0]
        last_dy = path[1][1] - path[0][1]
        for index in range(1, len(path) - 1):
            dx = path[index + 1][0] - path[index][0]
            dy = path[index + 1][1] - path[index][1]
            if (dx, dy) != (last_dx, last_dy):
                compressed.append(path[index])
                last_dx, last_dy = dx, dy
        compressed.append(path[-1])
        return compressed

    @staticmethod
    def _grid_path_is_free(
        path: list[tuple[int, int]],
        blocked: np.ndarray,
        *,
        allowed: set[tuple[int, int]],
    ) -> bool:
        for x, y in path:
            if (x, y) in allowed:
                continue
            if blocked[y, x] > 0:
                return False
        return True

    @staticmethod
    def _occupy_grid_path(blocked: np.ndarray, path: list[tuple[int, int]], *, padding: int) -> None:
        height, width = blocked.shape
        pad = max(0, int(padding))
        for x, y in path:
            x1 = max(0, x - pad)
            x2 = min(width, x + pad + 1)
            y1 = max(0, y - pad)
            y2 = min(height, y + pad + 1)
            blocked[y1:y2, x1:x2] = 1

    def _rasterize_grid_path(
        self,
        mask_u8: np.ndarray,
        path: list[tuple[int, int]],
        routing_state: dict[str, object],
        *,
        thickness: int,
    ) -> None:
        points = self._grid_path_to_pixels(path, routing_state)
        self._draw_polyline(mask_u8, points, thickness=max(1, int(thickness)))

    def _build_grid_path_mask(
        self,
        *,
        mask_shape: tuple[int, int],
        path: list[tuple[int, int]],
        routing_state: dict[str, object],
        thickness: int,
    ) -> np.ndarray:
        candidate_mask = np.zeros(mask_shape, dtype=np.uint8)
        self._rasterize_grid_path(candidate_mask, path, routing_state, thickness=thickness)
        return candidate_mask

    @staticmethod
    def _grid_path_to_pixels(path: list[tuple[int, int]], routing_state: dict[str, object]) -> list[tuple[int, int]]:
        xs = routing_state['xs']
        ys = routing_state['ys']
        return [(int(xs[x]), int(ys[y])) for x, y in path]

    def _place_ic_cell_blockages(
        self,
        routing_state: dict[str, object],
        centers: list[int],
        rng: random.Random,
    ) -> None:
        blocked = np.asarray(routing_state['blocked'], dtype=np.uint8)
        ys = routing_state['ys']
        grid_width = int(routing_state['grid_width'])
        for center in centers:
            gy = self._nearest_grid_y(center, ys)
            segment_count = max(2, min(5, grid_width // 6))
            for _ in range(segment_count):
                span = self._randint_safe(rng, 2, max(3, grid_width // 6))
                start_x = self._randint_safe(rng, 1, max(1, grid_width - span - 2))
                blocked[max(0, gy - 1): min(blocked.shape[0], gy + 2), start_x:start_x + span] = 1

    def _place_ic_channel_blockages(
        self,
        routing_state: dict[str, object],
        channel_centers: list[int],
        rng: random.Random,
    ) -> None:
        blocked = np.asarray(routing_state['blocked'], dtype=np.uint8)
        ys = routing_state['ys']
        grid_width = int(routing_state['grid_width'])
        for center in channel_centers:
            gy = self._nearest_grid_y(center, ys)
            obstacle_count = max(1, min(4, grid_width // 10))
            for _ in range(obstacle_count):
                span = self._randint_safe(rng, 1, max(2, grid_width // 8))
                start_x = self._randint_safe(rng, 1, max(2, grid_width - span - 3))
                obstacle_top = max(1, gy - rng.randint(1, 2))
                obstacle_bottom = min(blocked.shape[0] - 1, gy + rng.randint(1, 2))
                blocked[obstacle_top:obstacle_bottom + 1, start_x:start_x + span] = 1

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

    def _plan_ic_window_slots(
        self,
        width: int,
        count: int,
        rng: random.Random,
        *,
        style: str,
        margin: int,
    ) -> list[tuple[int, int]]:
        usable_left = int(max(2, margin))
        usable_right = int(max(usable_left + 12, width - margin - 1))
        usable_width = max(12, usable_right - usable_left)
        windows: list[tuple[int, int]] = []
        if style == 'cell_array':
            slot_count = max(2, min(int(count), max(2, usable_width // 18)))
            pitch = usable_width / float(slot_count)
            for slot_index in range(slot_count):
                center = usable_left + ((slot_index + 0.5) * pitch)
                center += rng.uniform(-pitch * 0.08, pitch * 0.08)
                slot_width = pitch * rng.uniform(0.52, 0.82)
                x1 = int(round(center - (slot_width / 2.0)))
                x2 = int(round(center + (slot_width / 2.0)))
                windows.append((max(usable_left, x1), min(usable_right, x2)))
        else:
            slot_count = max(2, min(int(count), 6))
            cursor = usable_left
            for _slot_index in range(slot_count * 2):
                if len(windows) >= slot_count or cursor >= usable_right - 12:
                    break
                cursor += rng.randint(4, max(5, usable_width // max(10, slot_count * 5)))
                slot_width = rng.randint(max(12, usable_width // 10), max(18, usable_width // max(4, slot_count)))
                x1 = cursor
                x2 = min(usable_right, x1 + slot_width)
                if x2 - x1 < 10:
                    break
                windows.append((x1, x2))
                cursor = x2
        sanitized: list[tuple[int, int]] = []
        last_end = usable_left - 1
        for x1, x2 in sorted(windows):
            clipped_x1 = max(usable_left, int(x1), last_end + 4)
            clipped_x2 = min(usable_right, int(x2))
            if clipped_x2 - clipped_x1 < 10:
                continue
            sanitized.append((clipped_x1, clipped_x2))
            last_end = clipped_x2
        return sanitized

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

    @staticmethod
    def _resolve_polyline_y_at_x(points: list[tuple[int, int]], x: int) -> int:
        if not points:
            return 0
        resolved_x = int(x)
        nearest_y = points[0][1]
        nearest_distance = abs(points[0][0] - resolved_x)
        for start_point, end_point in zip(points[:-1], points[1:]):
            if start_point[1] == end_point[1]:
                segment_min_x = min(start_point[0], end_point[0])
                segment_max_x = max(start_point[0], end_point[0])
                if segment_min_x <= resolved_x <= segment_max_x:
                    return int(start_point[1])
            for point_x, point_y in (start_point, end_point):
                distance = abs(point_x - resolved_x)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_y = point_y
        return int(nearest_y)

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

    def _decorate_terminals(
        self,
        mask_u8: np.ndarray,
        anchors: list[tuple[int, int]],
        family: str,
        rng: random.Random,
    ) -> np.ndarray:
        if int(np.count_nonzero(mask_u8)) <= 0 or not anchors:
            return mask_u8
        if self._domain() == 'ic':
            return self._decorate_ic_contacts(mask_u8, anchors)
        return self._decorate_pcb_terminals(mask_u8, anchors, family, rng)

    def _decorate_pcb_terminals(
        self,
        mask_u8: np.ndarray,
        anchors: list[tuple[int, int]],
        family: str,
        rng: random.Random,
    ) -> np.ndarray:
        updated = mask_u8.copy()
        pad_radius = max(2, self._width_range()[1] + 1)
        boundary_anchors = [anchor for anchor in anchors if self._is_boundary_anchor(anchor, updated.shape)]
        inner_anchors = [anchor for anchor in anchors if anchor not in boundary_anchors]
        if not boundary_anchors:
            boundary_anchors = anchors[:]
        rng.shuffle(boundary_anchors)
        rng.shuffle(inner_anchors)

        pad_count = max(2, min(len(boundary_anchors), max(2, int(round(self.params.trace_count * 0.35)))))
        for anchor in boundary_anchors[:pad_count]:
            cv2.circle(updated, anchor, pad_radius, 255, thickness=-1)

        via_count_range = self.params.via_count_range or self._default_via_count_range(family)
        via_min = max(0, int(via_count_range[0]))
        via_max = max(via_min, int(via_count_range[1]))
        via_target = 0 if via_max <= 0 else rng.randint(via_min, via_max)
        via_added = 0
        via_candidates = inner_anchors if inner_anchors else boundary_anchors
        for anchor in via_candidates:
            if via_added >= via_target:
                break
            outer_radius = max(3, pad_radius + rng.randint(0, 2))
            inner_radius = max(1, outer_radius // 2)
            if not self._ring_fits(updated, anchor, outer_radius):
                continue
            cv2.circle(updated, anchor, outer_radius, 255, thickness=-1)
            cv2.circle(updated, anchor, inner_radius, 0, thickness=-1)
            via_added += 1
        return updated

    def _decorate_ic_contacts(
        self,
        mask_u8: np.ndarray,
        anchors: list[tuple[int, int]],
    ) -> np.ndarray:
        updated = mask_u8.copy()
        bump_half = max(1, self._width_range()[1] + 1)
        if not anchors:
            return updated
        target_count = max(2, min(len(anchors), int(round(self.params.trace_count * 0.6))))
        if len(anchors) <= target_count:
            selected_anchors = anchors
        else:
            sample_indexes = np.linspace(0, len(anchors) - 1, target_count)
            selected_anchors = [anchors[int(round(index))] for index in sample_indexes]
        for anchor_x, anchor_y in selected_anchors:
            x1 = max(0, anchor_x - bump_half)
            y1 = max(0, anchor_y - bump_half)
            x2 = min(updated.shape[1], anchor_x + bump_half + 1)
            y2 = min(updated.shape[0], anchor_y + bump_half + 1)
            updated[y1:y2, x1:x2] = 255
        return updated

    def _default_via_count_range(self, family: str) -> tuple[int, int]:
        if self._domain() == 'ic':
            return (0, 0)
        if family == 'cell_array':
            upper = max(2, min(8, int(round(self.params.trace_count / 2.0))))
            return (2, upper)
        if family == 'tree':
            return (1, max(2, min(5, int(round(self.params.trace_count / 3.0)))))
        return (1, max(2, min(4, int(round(self.params.trace_count / 4.0)))))

    def _resolve_parallel_offsets(
        self,
        height: int,
        width: int,
        orientation: str,
        spacing: int,
        requested: int,
    ) -> list[int]:
        margin = self._effective_margin(height, width)
        if orientation == 'horizontal':
            axis_length = max(8, height - (margin * 2))
            count = max(1, min(requested, max(1, axis_length // max(1, spacing))))
            if count == 1:
                return [height // 2]
            start = margin + max(0, (axis_length - ((count - 1) * spacing)) // 2)
            return [int(np.clip(start + (index * spacing), margin, height - margin - 1)) for index in range(count)]
        if orientation == 'vertical':
            axis_length = max(8, width - (margin * 2))
            count = max(1, min(requested, max(1, axis_length // max(1, spacing))))
            if count == 1:
                return [width // 2]
            start = margin + max(0, (axis_length - ((count - 1) * spacing)) // 2)
            return [int(np.clip(start + (index * spacing), margin, width - margin - 1)) for index in range(count)]
        normal_limit = max(6, int(round((min(height, width) / 2.0) - margin)))
        count = max(1, min(requested, max(1, int((normal_limit * 2) // max(1, spacing)))))
        if count == 1:
            return [0]
        start = -normal_limit + max(0, ((normal_limit * 2) - ((count - 1) * spacing)) // 2)
        return [int(np.clip(start + (index * spacing), -normal_limit, normal_limit)) for index in range(count)]

    def _full_span_segment(
        self,
        height: int,
        width: int,
        orientation: str,
        offset: int,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if orientation == 'horizontal':
            y = int(np.clip(offset, 0, height - 1))
            return (0, y), (width - 1, y)
        if orientation == 'vertical':
            x = int(np.clip(offset, 0, width - 1))
            return (x, 0), (x, height - 1)
        angle_deg = 45.0 if orientation == 'diag_pos' else -45.0
        angle_rad = math.radians(angle_deg)
        direction = (math.cos(angle_rad), math.sin(angle_rad))
        normal = (-direction[1], direction[0])
        center_x = ((width - 1) / 2.0) + (normal[0] * float(offset))
        center_y = ((height - 1) / 2.0) + (normal[1] * float(offset))
        extent = max(width, height) * 2.0
        start = (
            int(round(center_x - (direction[0] * extent))),
            int(round(center_y - (direction[1] * extent))),
        )
        end = (
            int(round(center_x + (direction[0] * extent))),
            int(round(center_y + (direction[1] * extent))),
        )
        clipped = cv2.clipLine((0, 0, width, height), start, end)
        if not clipped[0]:
            return None
        return tuple(int(v) for v in clipped[1]), tuple(int(v) for v in clipped[2])

    def _random_parallel_offset(
        self,
        height: int,
        width: int,
        orientation: str,
        rng: random.Random,
    ) -> int:
        margin = self._effective_margin(height, width)
        if orientation == 'horizontal':
            return rng.randint(margin, max(margin + 1, height - margin - 1))
        if orientation == 'vertical':
            return rng.randint(margin, max(margin + 1, width - margin - 1))
        normal_limit = max(6, int(round((min(height, width) / 2.0) - margin)))
        return rng.randint(-normal_limit, normal_limit)

    def _build_axis_branch_endpoint(
        self,
        start_xy: tuple[int, int],
        *,
        length: int,
        direction: int,
        axis: str,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        if axis == 'x':
            return (
                int(np.clip(start_xy[0] + (int(direction) * int(length)), 0, width - 1)),
                int(start_xy[1]),
            )
        return (
            int(start_xy[0]),
            int(np.clip(start_xy[1] + (int(direction) * int(length)), 0, height - 1)),
        )

    def _build_pcb_diagonal_branch_endpoint(
        self,
        start_xy: tuple[int, int],
        *,
        width: int,
        height: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        diagonal_length = max(6, int(round(min(width, height) * rng.uniform(0.07, 0.18))))
        dx = diagonal_length if rng.random() < 0.5 else -diagonal_length
        dy = diagonal_length if rng.random() < 0.5 else -diagonal_length
        return (
            int(np.clip(start_xy[0] + dx, 0, width - 1)),
            int(np.clip(start_xy[1] + dy, 0, height - 1)),
        )

    def _resolve_branch_fractions(self, target: int) -> list[float]:
        branch_count = max(1, int(target))
        return [float(index + 1) / float(branch_count + 1) for index in range(branch_count)]

    @staticmethod
    def _point_on_segment(
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
        fraction: float,
    ) -> tuple[int, int]:
        fraction = float(min(max(fraction, 0.0), 1.0))
        return (
            int(round(start_xy[0] + ((end_xy[0] - start_xy[0]) * fraction))),
            int(round(start_xy[1] + ((end_xy[1] - start_xy[1]) * fraction))),
        )

    @staticmethod
    def _midpoint(start_xy: tuple[int, int], end_xy: tuple[int, int]) -> tuple[int, int]:
        return (
            int(round((start_xy[0] + end_xy[0]) / 2.0)),
            int(round((start_xy[1] + end_xy[1]) / 2.0)),
        )

    @staticmethod
    def _is_boundary_anchor(anchor_xy: tuple[int, int], shape_hw: tuple[int, int]) -> bool:
        x, y = anchor_xy
        height, width = shape_hw
        return x <= 1 or y <= 1 or x >= (width - 2) or y >= (height - 2)

    def _ring_fits(self, mask_u8: np.ndarray, center_xy: tuple[int, int], outer_radius: int) -> bool:
        center_x, center_y = center_xy
        height, width = mask_u8.shape
        x1 = max(0, center_x - outer_radius - 1)
        y1 = max(0, center_y - outer_radius - 1)
        x2 = min(width, center_x + outer_radius + 2)
        y2 = min(height, center_y + outer_radius + 2)
        if x2 <= x1 or y2 <= y1:
            return False
        return bool(np.count_nonzero(mask_u8[y1:y2, x1:x2]) > 0)

    @staticmethod
    def _allow_candidate_overlap(
        occupied_mask: np.ndarray,
        candidate_mask: np.ndarray,
        canvas_shape: tuple[int, int],
        allowed_points: list[tuple[int, int]],
        *,
        allowance_radius: int,
    ) -> bool:
        overlap = (occupied_mask > 0) & (candidate_mask > 0)
        if not np.any(overlap):
            return True
        allowed_mask = np.zeros(canvas_shape, dtype=np.uint8)
        radius = max(1, int(allowance_radius))
        for point_x, point_y in allowed_points:
            cv2.circle(allowed_mask, (int(point_x), int(point_y)), radius, 255, thickness=-1)
        return not np.any(overlap & (allowed_mask == 0))

    def _build_trace_polyline(
        self,
        height: int,
        width: int,
        rng: random.Random,
        *,
        segment_count: int,
    ) -> list[tuple[int, int]]:
        margin = self._effective_margin(height, width)
        min_x = margin
        max_x = max(margin + 1, width - margin - 1)
        min_y = margin
        max_y = max(margin + 1, height - margin - 1)

        current_x = rng.randint(min_x, max_x)
        current_y = rng.randint(min_y, max_y)
        horizontal = bool(rng.getrandbits(1))
        points: list[tuple[int, int]] = [(current_x, current_y)]
        segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        min_span = max(8, min(height, width) // 10)
        max_span = max(min_span + 2, min(height, width) // 2)

        for _segment_index in range(max(1, int(segment_count))):
            next_point: tuple[int, int] | None = None
            for _branch_attempt in range(16):
                segment_length = rng.randint(min_span, max_span)
                direction = -1 if rng.random() < 0.5 else 1
                next_x = current_x
                next_y = current_y
                if horizontal:
                    next_x = int(np.clip(current_x + direction * segment_length, min_x, max_x))
                else:
                    next_y = int(np.clip(current_y + direction * segment_length, min_y, max_y))
                if (next_x, next_y) == points[-1]:
                    continue
                candidate_segment = ((current_x, current_y), (next_x, next_y))
                if self._segment_intersects_existing(candidate_segment, segments):
                    continue
                next_point = (next_x, next_y)
                break
            if next_point is None:
                break
            points.append(next_point)
            segments.append(((current_x, current_y), next_point))
            current_x, current_y = next_point
            horizontal = not horizontal

        if len(points) == 1:
            fallback_x = int(np.clip(current_x + min_span, min_x, max_x))
            if fallback_x == current_x:
                fallback_x = int(np.clip(current_x - min_span, min_x, max_x))
            if fallback_x != current_x:
                candidate_segment = ((current_x, current_y), (fallback_x, current_y))
                if not self._segment_intersects_existing(candidate_segment, segments):
                    points.append((fallback_x, current_y))
        return points

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
        for start_point, end_point in zip(points[:-1], points[1:]):
            cv2.line(canvas, start_point, end_point, 255, thickness=resolved_thickness)

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

    def _render_ic_grayscale(self, mask_u8: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
        height, width = mask_u8.shape
        background_level = 0.18
        copper_base = 0.62
        blur_sigma = 0.55
        low_freq_scale = 0.022
        background = np.full((height, width), background_level, dtype=np.float32)
        low_freq = cv2.resize(
            np_rng.normal(0.0, 1.0, size=(max(2, height // 16), max(2, width // 16))).astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=2.0)
        low_freq /= max(1e-6, float(np.max(np.abs(low_freq))))
        background += low_freq * low_freq_scale

        copper = (mask_u8 > 0).astype(np.uint8)
        distance = cv2.distanceTransform(copper, cv2.DIST_L2, 5).astype(np.float32)
        if float(distance.max()) > 0.0:
            distance /= float(distance.max())
        copper_tone = copper_base + (distance * (0.14 if self._domain() == 'ic' else 0.18))
        image = background + copper.astype(np.float32) * copper_tone
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma)

        background_noise = np_rng.normal(
            0.0,
            float(self.params.background_noise_sigma),
            size=image.shape,
        ).astype(np.float32)
        trace_noise = np_rng.normal(
            0.0,
            float(self.params.trace_noise_sigma),
            size=image.shape,
        ).astype(np.float32) * copper.astype(np.float32)
        image += background_noise
        image += trace_noise
        return np.clip(image, 0.0, 1.0, out=image)

    def _render_pcb_rgb(self, mask_u8: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
        height, width = mask_u8.shape
        background_palette = np.asarray(
            np_rng.choice(
                np.asarray(
                    [
                        [0.34, 0.28, 0.14],
                        [0.29, 0.22, 0.10],
                        [0.20, 0.31, 0.13],
                        [0.25, 0.36, 0.16],
                    ],
                    dtype=np.float32,
                ),
                axis=0,
            ),
            dtype=np.float32,
        )
        copper_mode = str(np_rng.choice(['copper', 'dark', 'bright', 'through_mask']))
        copper_palette = {
            'copper': np.asarray([0.80, 0.47, 0.23], dtype=np.float32),
            'dark': np.asarray([0.50, 0.33, 0.18], dtype=np.float32),
            'bright': np.asarray([0.88, 0.70, 0.36], dtype=np.float32),
            'through_mask': (background_palette * 0.60) + np.asarray([0.20, 0.18, 0.08], dtype=np.float32),
        }
        copper_base = np.asarray(copper_palette[copper_mode], dtype=np.float32)

        background = np.ones((height, width, 3), dtype=np.float32) * background_palette[None, None, :]
        for channel_index in range(3):
            low_freq = cv2.resize(
                np_rng.normal(0.0, 1.0, size=(max(2, height // 18), max(2, width // 18))).astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=2.4)
            low_freq /= max(1e-6, float(np.max(np.abs(low_freq))))
            background[..., channel_index] += low_freq * float(np_rng.uniform(0.018, 0.045))

        copper = (mask_u8 > 0).astype(np.uint8)
        distance = cv2.distanceTransform(copper, cv2.DIST_L2, 5).astype(np.float32)
        if float(distance.max()) > 0.0:
            distance /= float(distance.max())
        copper_shade = 0.90 + (distance * 0.22)
        copper_rgb = copper_base[None, None, :] * copper_shade[..., None]
        image = background.copy()
        image[copper > 0] = copper_rgb[copper > 0]
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=0.75)

        background_noise = np_rng.normal(
            0.0,
            float(self.params.background_noise_sigma),
            size=image.shape,
        ).astype(np.float32)
        trace_noise = np_rng.normal(
            0.0,
            float(self.params.trace_noise_sigma),
            size=image.shape,
        ).astype(np.float32) * copper[..., None].astype(np.float32)
        image += background_noise
        image += trace_noise
        return np.clip(image, 0.0, 1.0, out=image)

    @staticmethod
    def _to_channel_first(image: np.ndarray, channels: int) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32)
        resolved_channels = max(1, int(channels))
        if array.ndim == 2:
            base = array[None, :, :]
            if resolved_channels == 1:
                return base.astype(np.float32, copy=False)
            return np.repeat(base, resolved_channels, axis=0).astype(np.float32, copy=False)
        if array.ndim != 3 or array.shape[2] <= 0:
            raise ValueError('SyntheticTopologyGenerator expects 2D grayscale or HWC RGB image.')
        if resolved_channels == 1:
            grayscale = (
                (array[..., 0] * 0.299)
                + (array[..., 1] * 0.587)
                + (array[..., 2] * 0.114)
            )
            return grayscale[None, :, :].astype(np.float32, copy=False)
        chw = np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)
        if resolved_channels == chw.shape[0]:
            return chw
        if resolved_channels < chw.shape[0]:
            return chw[:resolved_channels].astype(np.float32, copy=False)
        repeats = [chw]
        while sum(item.shape[0] for item in repeats) < resolved_channels:
            repeats.append(chw[: min(chw.shape[0], resolved_channels - sum(item.shape[0] for item in repeats))])
        return np.concatenate(repeats, axis=0).astype(np.float32, copy=False)
