"""repository: point uncertainty cluster fraction."""
from __future__ import annotations
import math
import numpy as np

def _point_uncertainty_cluster_fraction(
    coordinates: tuple[tuple[float, float, float], ...],
    uncertainty_values: np.ndarray,
    support_mask: np.ndarray,
    *,
    uncertainty_threshold: float,
) -> float:
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    uncertainties = np.asarray(uncertainty_values, dtype=np.float64).reshape(-1)
    if not coordinates or support.size == 0 or not np.any(support):
        return 0.0
    uncertain_indices = [
        index
        for index, (is_supported, uncertainty) in enumerate(zip(support.tolist(), uncertainties.tolist()))
        if is_supported and np.isfinite(uncertainty) and float(uncertainty) > float(uncertainty_threshold)
    ]
    if not uncertain_indices:
        return 0.0
    support_count = int(np.count_nonzero(support))
    adjacency: dict[int, list[int]] = {index: [] for index in uncertain_indices}
    for offset, left_index in enumerate(uncertain_indices):
        left_x, left_y, left_radius = coordinates[left_index]
        left_radius = max(1.0, float(left_radius))
        for right_index in uncertain_indices[offset + 1:]:
            right_x, right_y, right_radius = coordinates[right_index]
            right_radius = max(1.0, float(right_radius))
            max_distance = left_radius + right_radius + 1.0
            if math.hypot(float(left_x) - float(right_x), float(left_y) - float(right_y)) <= max_distance:
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)
    visited: set[int] = set()
    largest_cluster = 0
    for start_index in uncertain_indices:
        if start_index in visited:
            continue
        stack = [start_index]
        visited.add(start_index)
        cluster_size = 0
        while stack:
            current = stack.pop()
            cluster_size += 1
            for neighbor in adjacency.get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        largest_cluster = max(largest_cluster, cluster_size)
    return float(largest_cluster / max(1, support_count))
