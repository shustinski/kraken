"""repository: point labeled quality."""
from __future__ import annotations
import math
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POINT_SUPERVISED_SCORE_WEIGHTS,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PointLabeledMetrics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _weighted_mean

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.match_point_sets import (
    _match_point_sets,
    _point_count_error,
    _point_distance_scores,
    _point_match_threshold,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.paint_disk import _paint_disk

def _point_labeled_error_map(prediction_view: object, gt_view: object, point_match_radius: float) -> np.ndarray:
    shape = tuple(int(v) for v in getattr(prediction_view, 'pred_gray').shape)
    result = np.zeros(shape, dtype=np.float32)
    pred_points = tuple(getattr(prediction_view, 'points', ()))
    gt_points = tuple(getattr(gt_view, 'points', ()))
    matched_distances, matched_pred, matched_gt = _match_point_sets(pred_points, gt_points, point_match_radius)
    diagonal = float(max(EPS, math.hypot(float(shape[0]), float(shape[1]))))
    matched_pairs: list[tuple[int, int, float]] = []
    candidate_pairs: list[tuple[float, int, int]] = []
    for index_a, point_a in enumerate(pred_points):
        for index_b, point_b in enumerate(gt_points):
            distance = float(np.hypot(float(getattr(point_a, 'x', 0.0)) - float(getattr(point_b, 'x', 0.0)), float(getattr(point_a, 'y', 0.0)) - float(getattr(point_b, 'y', 0.0))))
            if distance <= _point_match_threshold(point_a, point_b, point_match_radius):
                candidate_pairs.append((distance, index_a, index_b))
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    for distance, index_a, index_b in candidate_pairs:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matched_pairs.append((index_a, index_b, distance))
    for index_a, index_b, distance in matched_pairs:
        severity = float(min(1.0, distance / max(EPS, point_match_radius, diagonal * 0.05)))
        point_a = pred_points[index_a]
        point_b = gt_points[index_b]
        _paint_disk(result, float(getattr(point_a, 'x', 0.0)), float(getattr(point_a, 'y', 0.0)), max(1.0, float(getattr(point_a, 'radius', 0.0))) + 1.0, severity)
        _paint_disk(result, float(getattr(point_b, 'x', 0.0)), float(getattr(point_b, 'y', 0.0)), max(1.0, float(getattr(point_b, 'radius', 0.0))) + 1.0, severity)
    for index, point in enumerate(pred_points):
        if index not in matched_pred:
            _paint_disk(result, float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), max(1.0, float(getattr(point, 'radius', 0.0))) + 1.0, 1.0)
    for index, point in enumerate(gt_points):
        if index not in matched_gt:
            _paint_disk(result, float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), max(1.0, float(getattr(point, 'radius', 0.0))) + 1.0, 0.85)
    return np.clip(result, 0.0, 1.0)

def _point_labeled_quality(prediction_view: object, gt_view: object, point_match_radius: float) -> PointLabeledMetrics:
    points_pred = tuple(getattr(prediction_view, "points", ()))
    points_gt = tuple(getattr(gt_view, "points", ()))
    matched_distances, matched_pred, matched_gt = _match_point_sets(points_pred, points_gt, point_match_radius)
    predicted_count = len(points_pred)
    target_count = len(points_gt)
    matched_count = len(matched_distances)
    if predicted_count == 0 and target_count == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = float(matched_count / max(1, predicted_count))
        recall = float(matched_count / max(1, target_count))
    f1 = 0.0 if precision + recall <= EPS else float((2.0 * precision * recall) / (precision + recall))
    mean_localization_error = float(np.mean(np.asarray(matched_distances, dtype=np.float64))) if matched_count > 0 else 0.0
    if matched_count <= 0:
        localization_score = 1.0 if predicted_count == 0 and target_count == 0 else 0.0
    else:
        localization_score = float(1.0 - min(1.0, mean_localization_error / max(EPS, float(point_match_radius))))
    chamfer_distance, hausdorff_distance = _point_distance_scores(points_pred, points_gt, tuple(int(v) for v in prediction_view.pred_gray.shape))
    diagonal = float(max(EPS, math.hypot(float(prediction_view.pred_gray.shape[0]), float(prediction_view.pred_gray.shape[1]))))
    chamfer_score = float(1.0 - min(1.0, chamfer_distance / diagonal))
    hausdorff_score = float(1.0 - min(1.0, hausdorff_distance / diagonal))
    count_error = _point_count_error(predicted_count, target_count)
    quality = _weighted_mean([
        (f1, POINT_SUPERVISED_SCORE_WEIGHTS["f1"]),
        (localization_score, POINT_SUPERVISED_SCORE_WEIGHTS["localization"]),
        (1.0 - count_error, POINT_SUPERVISED_SCORE_WEIGHTS["count_term"]),
    ])
    return PointLabeledMetrics(
        precision_at_radius=float(precision),
        recall_at_radius=float(recall),
        f1_at_radius=float(f1),
        mean_localization_error=float(mean_localization_error),
        localization_score=float(localization_score),
        chamfer_score=float(chamfer_score),
        hausdorff_score=float(hausdorff_score),
        count_error=float(count_error),
        matched_count=int(matched_count),
        predicted_count=int(predicted_count),
        target_count=int(target_count),
        quality_score=float(quality),
        error_score=float(1.0 - quality),
    )
