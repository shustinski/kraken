"""synthetic_topology: place ic channel blockages."""
from __future__ import annotations
import random
import numpy as np


class PlaceIcChannelBlockagesMixin:
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
