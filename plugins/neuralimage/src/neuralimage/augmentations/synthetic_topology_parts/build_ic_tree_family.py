"""synthetic_topology: build ic tree family."""
from __future__ import annotations
import random
import numpy as np


class BuildIcTreeFamilyMixin:
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
