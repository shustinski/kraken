"""repository: mask structure."""
from __future__ import annotations
import math
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _clip01

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _boundary_mask,
    _component_area_stats,
    _distance_transform,
    _label_components,
    _neighbor_count,
    skeletonize,
)

def _mask_structure(
    mask: np.ndarray,
    *,
    include_skeleton: bool = True,
    include_boundary_distance: bool = False,
) -> dict[str, object]:
    from neuralimage.Validation_gradient_widget_lite.core.repository import cv2, ndi

    mask_bool = np.asarray(mask, dtype=bool)
    labels, count = _label_components(mask_bool)
    boundary = _boundary_mask(mask_bool)
    boundary_dist = _distance_transform(~boundary) if include_boundary_distance and np.any(boundary) and (ndi is not None or cv2 is not None) else None
    if include_skeleton:
        skeleton = skeletonize(mask_bool)
        if np.any(skeleton):
            skeleton_neighbors = _neighbor_count(skeleton)
            endpoint_count = int(np.count_nonzero(skeleton & (skeleton_neighbors == 1)))
            branchpoint_count = int(np.count_nonzero(skeleton & (skeleton_neighbors >= 3)))
        else:
            skeleton_neighbors = np.zeros_like(mask_bool, dtype=np.uint8)
            endpoint_count = 0
            branchpoint_count = 0
    else:
        skeleton = np.zeros_like(mask_bool, dtype=bool)
        skeleton_neighbors = np.zeros_like(mask_bool, dtype=np.uint8)
        endpoint_count = 0
        branchpoint_count = 0
    _component_areas, mean_component_area = _component_area_stats(labels, count)
    area = float(np.count_nonzero(mask_bool))
    return {
        "labels": labels,
        "component_count": int(count),
        "area_fraction": float(area / max(1, mask_bool.size)),
        "mean_component_area": mean_component_area,
        "has_skeleton": bool(include_skeleton),
        "boundary": boundary,
        "boundary_dist": boundary_dist,
        "skeleton": skeleton,
        "skeleton_neighbors": skeleton_neighbors,
        "skeleton_length": float(np.count_nonzero(skeleton)),
        "endpoint_count": endpoint_count,
        "branchpoint_count": branchpoint_count,
    }

def _tiny_component_map_from_structure(structure: dict[str, object], area_threshold: int = 12) -> np.ndarray:
    labels = np.asarray(structure["labels"], dtype=np.int32)
    count = int(structure["component_count"])
    result = np.zeros(labels.shape, dtype=np.float32)
    if count <= 0:
        return result
    area_counts = np.bincount(labels.ravel(), minlength=count + 1)
    small_labels = np.flatnonzero(area_counts[1:count + 1] <= int(area_threshold)) + 1
    for label_id in small_labels:
        result[labels == int(label_id)] = 1.0
    return result

def _tiny_component_map(mask: np.ndarray, area_threshold: int = 12) -> np.ndarray:
    return _tiny_component_map_from_structure(_mask_structure(mask), area_threshold=area_threshold)

def _thin_bridge_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    if not np.any(skeleton):
        return np.zeros_like(skeleton, dtype=np.float32)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    bridge = skeleton & (neighbors == 2)
    return np.asarray(_binary_dilate(bridge, radius=1), dtype=np.float32)

def _thin_bridge_map(mask: np.ndarray) -> np.ndarray:
    return _thin_bridge_map_from_structure(_mask_structure(mask))

def _branchpoint_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    return np.asarray(_binary_dilate(skeleton & (neighbors >= 3), radius=1), dtype=np.float32)

def _branchpoint_map(mask: np.ndarray) -> np.ndarray:
    return _branchpoint_map_from_structure(_mask_structure(mask))

def _endpoint_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    return np.asarray(_binary_dilate(skeleton & (neighbors == 1), radius=1), dtype=np.float32)

def _endpoint_map(mask: np.ndarray) -> np.ndarray:
    return _endpoint_map_from_structure(_mask_structure(mask))

def _boundary_f1(pred: np.ndarray, gt: np.ndarray, radius: int = 1) -> float:
    pred_boundary = _boundary_mask(pred)
    gt_boundary = _boundary_mask(gt)
    if not np.any(pred_boundary) and not np.any(gt_boundary):
        return 1.0
    pred_d = _binary_dilate(pred_boundary, radius=radius)
    gt_d = _binary_dilate(gt_boundary, radius=radius)
    precision = float(np.count_nonzero(pred_boundary & gt_d) / max(1, np.count_nonzero(pred_boundary)))
    recall = float(np.count_nonzero(gt_boundary & pred_d) / max(1, np.count_nonzero(gt_boundary)))
    if precision + recall <= EPS:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))

def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    intersection = int(np.count_nonzero(pred_bool & gt_bool))
    total = int(np.count_nonzero(pred_bool)) + int(np.count_nonzero(gt_bool))
    if total == 0:
        return 1.0
    return float((2.0 * intersection + EPS) / (total + EPS))

def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    intersection = int(np.count_nonzero(pred_bool & gt_bool))
    union = int(np.count_nonzero(pred_bool | gt_bool))
    if union == 0:
        return 1.0
    return float((intersection + EPS) / (union + EPS))

def _precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    tp = int(np.count_nonzero(pred_bool & gt_bool))
    fp = int(np.count_nonzero(pred_bool & np.logical_not(gt_bool)))
    return float((tp + EPS) / (tp + fp + EPS))

def _recall(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    tp = int(np.count_nonzero(pred_bool & gt_bool))
    fn = int(np.count_nonzero(np.logical_not(pred_bool) & gt_bool))
    return float((tp + EPS) / (tp + fn + EPS))

def _soft_dice(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.clip(np.asarray(first, dtype=np.float32), 0.0, 1.0)
    second_arr = np.clip(np.asarray(second, dtype=np.float32), 0.0, 1.0)
    intersection = float(np.sum(first_arr * second_arr, dtype=np.float64))
    total = float(np.sum(first_arr, dtype=np.float64) + np.sum(second_arr, dtype=np.float64))
    if total <= EPS:
        return 1.0
    return float((2.0 * intersection + EPS) / (total + EPS))

def _soft_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.clip(np.asarray(first, dtype=np.float32), 0.0, 1.0)
    second_arr = np.clip(np.asarray(second, dtype=np.float32), 0.0, 1.0)
    intersection = float(np.sum(first_arr * second_arr, dtype=np.float64))
    union = float(np.sum(first_arr, dtype=np.float64) + np.sum(second_arr, dtype=np.float64) - intersection)
    if union <= EPS:
        return 1.0
    return float((intersection + EPS) / (union + EPS))

def _ssim_score(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.clip(np.asarray(first, dtype=np.float64), 0.0, 1.0)
    second_arr = np.clip(np.asarray(second, dtype=np.float64), 0.0, 1.0)
    if first_arr.size == 0 and second_arr.size == 0:
        return 1.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_first = float(first_arr.mean())
    mu_second = float(second_arr.mean())
    sigma_first = float(np.mean((first_arr - mu_first) ** 2, dtype=np.float64))
    sigma_second = float(np.mean((second_arr - mu_second) ** 2, dtype=np.float64))
    sigma_cross = float(np.mean((first_arr - mu_first) * (second_arr - mu_second), dtype=np.float64))
    denominator = (mu_first ** 2 + mu_second ** 2 + c1) * (sigma_first + sigma_second + c2)
    if denominator <= EPS:
        return 1.0 if np.allclose(first_arr, second_arr) else 0.0
    numerator = (2.0 * mu_first * mu_second + c1) * (2.0 * sigma_cross + c2)
    return float(_clip01(numerator / denominator))

def _mae(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.asarray(first, dtype=np.float32)
    second_arr = np.asarray(second, dtype=np.float32)
    return float(np.mean(np.abs(first_arr - second_arr), dtype=np.float64)) if first_arr.size else 0.0

def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.asarray(first, dtype=np.float32)
    second_arr = np.asarray(second, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(first_arr - second_arr), dtype=np.float64))) if first_arr.size else 0.0

def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size <= 0:
        return None
    return float(cols.mean(dtype=np.float64)), float(rows.mean(dtype=np.float64))

def _frame_diagonal(shape: tuple[int, int]) -> float:
    return float(max(EPS, math.hypot(float(shape[0]), float(shape[1]))))

def _centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_center = _mask_centroid(first)
    second_center = _mask_centroid(second)
    diagonal = _frame_diagonal(tuple(int(v) for v in np.asarray(first).shape))
    if first_center is None and second_center is None:
        return 0.0
    if first_center is None or second_center is None:
        return diagonal
    return float(math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))

def _distance_similarity(distance: float, shape: tuple[int, int]) -> float:
    return float(1.0 - min(1.0, float(distance) / _frame_diagonal(shape)))
