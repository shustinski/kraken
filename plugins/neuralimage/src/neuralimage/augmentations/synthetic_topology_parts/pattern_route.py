"""synthetic_topology: pattern route."""
from __future__ import annotations
import random
import numpy as np


class PatternRouteMixin:
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
