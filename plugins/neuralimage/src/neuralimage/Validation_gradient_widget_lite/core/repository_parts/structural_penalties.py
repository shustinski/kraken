"""repository: structural penalties."""
from __future__ import annotations
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _boundary_mask,
    _distance_transform,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import (
    _frame_diagonal,
    _mask_structure,
)

def _nearest_distances_between_coordinate_sets(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords_a = np.asarray(coords_a, dtype=np.float32)
    coords_b = np.asarray(coords_b, dtype=np.float32)
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return np.zeros(coords_a.shape[0], dtype=np.float64), np.zeros(coords_b.shape[0], dtype=np.float64)

    def _chunked_nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        pair_budget = 1_000_000
        chunk_size = max(1, min(int(source.shape[0]), int(max(1, pair_budget // max(1, int(target.shape[0]))))))
        nearest = np.empty(source.shape[0], dtype=np.float64)
        for start in range(0, source.shape[0], chunk_size):
            stop = min(source.shape[0], start + chunk_size)
            diff = np.asarray(source[start:stop, None, :] - target[None, :, :], dtype=np.float32)
            squared = np.sum(np.square(diff, dtype=np.float32), axis=2, dtype=np.float64)
            nearest[start:stop] = np.sqrt(np.min(squared, axis=1, initial=np.inf))
        return nearest

    if cKDTree is not None:
        tree_a = cKDTree(coords_a)
        tree_b = cKDTree(coords_b)
        nearest_a = np.asarray(tree_b.query(coords_a, k=1)[0], dtype=np.float64)
        nearest_b = np.asarray(tree_a.query(coords_b, k=1)[0], dtype=np.float64)
        return nearest_a, nearest_b
    return _chunked_nearest_distances(coords_a, coords_b), _chunked_nearest_distances(coords_b, coords_a)

def _hausdorff_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_structure: dict[str, object] | None = None,
    second_structure: dict[str, object] | None = None,
) -> float:
    first_boundary = np.asarray((first_structure or {}).get("boundary"), dtype=bool) if (first_structure or {}).get("boundary") is not None else _boundary_mask(first)
    second_boundary = np.asarray((second_structure or {}).get("boundary"), dtype=bool) if (second_structure or {}).get("boundary") is not None else _boundary_mask(second)
    shape = tuple(int(v) for v in np.asarray(first_boundary).shape)
    diagonal = _frame_diagonal(shape)
    if not np.any(first_boundary) and not np.any(second_boundary):
        return 0.0
    if not np.any(first_boundary) or not np.any(second_boundary):
        return diagonal
    if ndi is not None or cv2 is not None:
        dist_to_second = np.asarray((second_structure or {}).get("boundary_dist"), dtype=np.float32) if (second_structure or {}).get("boundary_dist") is not None else _distance_transform(~second_boundary)
        dist_to_first = np.asarray((first_structure or {}).get("boundary_dist"), dtype=np.float32) if (first_structure or {}).get("boundary_dist") is not None else _distance_transform(~first_boundary)
        directed_first = float(np.max(dist_to_second[first_boundary])) if np.any(first_boundary) else 0.0
        directed_second = float(np.max(dist_to_first[second_boundary])) if np.any(second_boundary) else 0.0
        return float(max(directed_first, directed_second))
    first_points = np.argwhere(first_boundary)
    second_points = np.argwhere(second_boundary)
    if first_points.size == 0 and second_points.size == 0:
        return 0.0
    if first_points.size == 0 or second_points.size == 0:
        return diagonal
    nearest_first, nearest_second = _nearest_distances_between_coordinate_sets(first_points, second_points)
    directed_first = float(np.max(nearest_first, initial=0.0))
    directed_second = float(np.max(nearest_second, initial=0.0))
    return float(max(directed_first, directed_second))

def _structural_penalties(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    pred_structure: dict[str, object] | None = None,
    gt_structure: dict[str, object] | None = None,
) -> tuple[float, int, int, float]:
    current_pred_structure = pred_structure or _mask_structure(pred)
    current_gt_structure = gt_structure or _mask_structure(gt)
    gt_count = int(current_gt_structure["component_count"])
    pred_count = int(current_pred_structure["component_count"])
    delta_cc = float(abs(pred_count - gt_count) / max(1, gt_count))
    if not bool(current_pred_structure.get("has_skeleton", True)) or not bool(current_gt_structure.get("has_skeleton", True)):
        return delta_cc, 0, 0, 0.0

    skeleton_gt = np.asarray(current_gt_structure["skeleton"], dtype=bool)
    supported_gt = skeleton_gt & _binary_dilate(pred, radius=1)
    _supported_labels, supported_count = _label_components(supported_gt)
    _gt_skel_labels, gt_skel_count = _label_components(skeleton_gt)
    break_count = max(0, int(supported_count - gt_skel_count))

    pred_labels = np.asarray(current_pred_structure["labels"], dtype=np.int32)
    false_bridge_count = 0
    if pred_count > 0 and gt_count > 1:
        expanded_gt = _binary_dilate(gt, radius=1)
        expanded_gt_labels, _ = _label_components(expanded_gt)
        for label_id in range(1, pred_count + 1):
            component_mask = pred_labels == label_id
            touched = set(int(value) for value in np.unique(expanded_gt_labels[component_mask]) if int(value) > 0)
            if len(touched) >= 2:
                false_bridge_count += len(touched) - 1

    length_gt = float(current_gt_structure["skeleton_length"])
    length_pred = float(current_pred_structure["skeleton_length"])
    skeleton_delta = float(abs(length_pred - length_gt) / max(1.0, length_gt))
    return delta_cc, int(break_count), int(false_bridge_count), skeleton_delta
