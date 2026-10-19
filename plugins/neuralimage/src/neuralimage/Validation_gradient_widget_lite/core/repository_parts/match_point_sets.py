"""repository: match point sets."""
from __future__ import annotations
import math
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    BCE_SCORE_CAP,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _clip01

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.structural_penalties import _nearest_distances_between_coordinate_sets

def _point_coordinates(points: tuple[object, ...]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([[float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0))] for point in points], dtype=np.float32)

def _point_match_threshold(point_a: object, point_b: object, base_radius: float) -> float:
    _ = point_a, point_b
    return float(max(0.0, float(base_radius)))

def _match_point_sets(points_a: tuple[object, ...], points_b: tuple[object, ...], base_radius: float) -> tuple[list[float], set[int], set[int]]:
    from neuralimage.Validation_gradient_widget_lite.core.repository import cKDTree

    candidate_pairs: list[tuple[float, int, int]] = []
    if cKDTree is not None and points_a and points_b:
        coords_a = _point_coordinates(points_a)
        coords_b = _point_coordinates(points_b)
        radius = max(0.0, float(base_radius))
        if coords_a.shape[0] > 0 and coords_b.shape[0] > 0 and radius >= 0.0:
            tree_b = cKDTree(coords_b)
            neighbors = tree_b.query_ball_point(coords_a, r=radius)
            for index_a, neighbor_indices in enumerate(neighbors):
                point_a = points_a[index_a]
                for index_b in neighbor_indices:
                    point_b = points_b[int(index_b)]
                    distance = float(np.hypot(
                        float(getattr(point_a, "x", 0.0)) - float(getattr(point_b, "x", 0.0)),
                        float(getattr(point_a, "y", 0.0)) - float(getattr(point_b, "y", 0.0)),
                    ))
                    if distance <= _point_match_threshold(point_a, point_b, base_radius):
                        candidate_pairs.append((distance, index_a, int(index_b)))
    else:
        for index_a, point_a in enumerate(points_a):
            for index_b, point_b in enumerate(points_b):
                distance = float(np.hypot(float(getattr(point_a, "x", 0.0)) - float(getattr(point_b, "x", 0.0)), float(getattr(point_a, "y", 0.0)) - float(getattr(point_b, "y", 0.0))))
                if distance <= _point_match_threshold(point_a, point_b, base_radius):
                    candidate_pairs.append((distance, index_a, index_b))
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_distances: list[float] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for distance, index_a, index_b in candidate_pairs:
        if index_a in matched_a or index_b in matched_b:
            continue
        matched_a.add(index_a)
        matched_b.add(index_b)
        matched_distances.append(float(distance))
    return matched_distances, matched_a, matched_b

def _point_distance_scores(points_a: tuple[object, ...], points_b: tuple[object, ...], frame_shape: tuple[int, int]) -> tuple[float, float]:
    coords_a = _point_coordinates(points_a)
    coords_b = _point_coordinates(points_b)
    if coords_a.shape[0] == 0 and coords_b.shape[0] == 0:
        return 0.0, 0.0
    diagonal = float(max(EPS, math.hypot(float(frame_shape[0]), float(frame_shape[1]))))
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return diagonal, diagonal
    nearest_a, nearest_b = _nearest_distances_between_coordinate_sets(coords_a, coords_b)
    chamfer = float((nearest_a.mean(dtype=np.float64) + nearest_b.mean(dtype=np.float64)) / 2.0)
    hausdorff = float(max(float(nearest_a.max(initial=0.0)), float(nearest_b.max(initial=0.0))))
    return chamfer, hausdorff

def _point_count_error(predicted_count: int, target_count: int) -> float:
    return float(_clip01(abs(int(predicted_count) - int(target_count)) / max(1.0, float(target_count))))

def _point_count_agreement(first_count: int, second_count: int) -> float:
    return float(1.0 - _clip01(abs(int(first_count) - int(second_count)) / max(1.0, float(max(first_count, second_count, 1)))))

def _mask_count_agreement(first_count: int, second_count: int) -> float:
    return float(1.0 - _clip01(abs(int(first_count) - int(second_count)) / max(1.0, float(max(first_count, second_count, 1)))))

def _symmetric_binary_cross_entropy(first_prob: np.ndarray, second_prob: np.ndarray) -> float:
    """Compute symmetric BCE between two probability maps."""

    first = np.clip(np.asarray(first_prob, dtype=np.float64), EPS, 1.0 - EPS)
    second = np.clip(np.asarray(second_prob, dtype=np.float64), EPS, 1.0 - EPS)
    forward = -(first * np.log(second) + (1.0 - first) * np.log(1.0 - second))
    backward = -(second * np.log(first) + (1.0 - second) * np.log(1.0 - first))
    return float(np.mean((forward + backward) * 0.5, dtype=np.float64)) if forward.size else 0.0

def _polygon_bce_score(bce_value: float) -> float:
    capped = min(float(bce_value) / max(EPS, float(BCE_SCORE_CAP)), 1.0)
    return float(max(0.0, 100.0 * (1.0 - capped)))
