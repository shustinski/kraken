"""repository: point internal confidence."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POINT_SUPPORT_THRESHOLD,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PointConfidenceMetrics,
    PointObjectConfidence,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.frame_uncertainty_components_points import _frame_uncertainty_components_from_points

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.internal_confidence_probability_map import (
    _confidence_map_from_probability,
    _internal_confidence_probability_map,
    _prob_from_gray,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.point_local_contrast import (
    _point_confidence_patch,
    _point_local_contrast,
)

def _point_internal_confidence(prediction_view: object, *, neighborhood_radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS, include_objects: bool = True) -> PointConfidenceMetrics:
    probability = _internal_confidence_probability_map(
        _prob_from_gray(np.asarray(getattr(prediction_view, 'pred_gray'), dtype=np.uint8)),
        support_mask=np.asarray(getattr(prediction_view, 'pred_bin'), dtype=bool),
    )
    confidence_map = _confidence_map_from_probability(probability)
    points = tuple(getattr(prediction_view, 'points', ()))
    if not points:
        return PointConfidenceMetrics(
            frame_uncertainty_score=0.0,
            mean_uncertainty=0.0,
            uncertain_support_fraction=0.0,
            top_uncertainty_mean=0.0,
            largest_uncertain_region_fraction=0.0,
            mean_point_confidence=0.0,
            mean_center_confidence=0.0,
            mean_local_confidence=0.0,
            mean_point_probability=0.0,
            mean_point_contrast=0.0,
            point_count=0,
            low_conf_fraction=0.0,
            worst_tail_uncertainty=0.0,
            largest_low_conf_component=0.0,
            objects=(),
        )
    object_rows: list[PointObjectConfidence] = []
    center_confidences: list[float] = []
    local_confidences: list[float] = []
    point_probs: list[float] = []
    point_contrasts: list[float] = []
    point_weights: list[float] = []
    point_coordinates: list[tuple[float, float, float]] = []
    local_radius = max(0, int(neighborhood_radius))
    for index, point in enumerate(points, start=1):
        x = float(getattr(point, 'x', 0.0))
        y = float(getattr(point, 'y', 0.0))
        px = int(round(x))
        py = int(round(y))
        if py < 0 or py >= probability.shape[0] or px < 0 or px >= probability.shape[1]:
            point_prob = 0.0
            center_confidence = 0.0
        else:
            point_prob = float(probability[py, px])
            center_confidence = float(confidence_map[py, px])
        local_patch = _point_confidence_patch(confidence_map, x, y, local_radius)
        local_confidence = float(np.mean(local_patch, dtype=np.float64)) if local_patch.size > 0 else center_confidence
        local_contrast = _point_local_contrast(probability, x, y, radius=local_radius)
        radius = max(1.0, float(getattr(point, 'radius', 0.0)))
        if include_objects:
            object_rows.append(PointObjectConfidence(
                object_id=int(index),
                x=x,
                y=y,
                radius=radius,
                point_probability=point_prob,
                center_confidence=center_confidence,
                local_confidence=local_confidence,
                local_contrast=local_contrast,
            ))
        center_confidences.append(center_confidence)
        local_confidences.append(local_confidence)
        point_probs.append(point_prob)
        point_contrasts.append(local_contrast)
        point_weights.append(float(max(0.0, min(1.0, (point_prob - float(POINT_SUPPORT_THRESHOLD)) / max(EPS, 1.0 - float(POINT_SUPPORT_THRESHOLD))))) if point_prob >= float(POINT_SUPPORT_THRESHOLD) else 0.0)
        point_coordinates.append((x, y, radius))
    confidence_array = np.asarray(center_confidences, dtype=np.float64)
    weight_array = np.asarray(point_weights, dtype=np.float64)
    weight_sum = float(np.sum(weight_array, dtype=np.float64))
    frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = _frame_uncertainty_components_from_points(
        np.asarray(point_probs, dtype=np.float32),
        tuple(point_coordinates),
        support_threshold=POINT_SUPPORT_THRESHOLD,
    )
    return PointConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        mean_uncertainty=mean_uncertainty,
        uncertain_support_fraction=low_conf_fraction,
        top_uncertainty_mean=worst_tail_uncertainty,
        largest_uncertain_region_fraction=largest_low_conf_component,
        mean_point_confidence=float(np.sum(confidence_array * weight_array, dtype=np.float64) / max(EPS, weight_sum)) if weight_sum > 0.0 else 0.0,
        mean_center_confidence=float(np.mean(confidence_array, dtype=np.float64)),
        mean_local_confidence=float(np.mean(np.asarray(local_confidences, dtype=np.float64))),
        mean_point_probability=float(np.mean(np.asarray(point_probs, dtype=np.float64))),
        mean_point_contrast=float(np.mean(np.asarray(point_contrasts, dtype=np.float64))),
        point_count=int(len(points)),
        low_conf_fraction=low_conf_fraction,
        worst_tail_uncertainty=worst_tail_uncertainty,
        largest_low_conf_component=largest_low_conf_component,
        objects=tuple(object_rows),
    )
