"""repository: point model agreement."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POINT_AGREEMENT_SCORE_WEIGHTS,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PointAgreementMetrics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _weighted_mean

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _boundary_mask,
    _distance_transform,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import (
    _mask_centroid,
    _mask_structure,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.match_point_sets import (
    _match_point_sets,
    _point_count_agreement,
)

def _point_model_agreement(first_view: object, second_view: object, point_match_radius: float) -> PointAgreementMetrics:
    points_a = tuple(getattr(first_view, "points", ()))
    points_b = tuple(getattr(second_view, "points", ()))
    matched_distances, _matched_a, _matched_b = _match_point_sets(points_a, points_b, point_match_radius)
    count_a = len(points_a)
    count_b = len(points_b)
    matched_count = len(matched_distances)
    false_positive_count = max(0, int(count_a - matched_count))
    false_negative_count = max(0, int(count_b - matched_count))
    if count_a == 0 and count_b == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = float(matched_count / max(1, count_a))
        recall = float(matched_count / max(1, count_b))
    f1 = 0.0 if precision + recall <= EPS else float((2.0 * precision * recall) / (precision + recall))
    mean_localization_error = float(np.mean(np.asarray(matched_distances, dtype=np.float64))) if matched_count > 0 else 0.0
    if matched_count <= 0:
        localization_agreement = 1.0 if count_a == 0 and count_b == 0 else 0.0
    else:
        localization_agreement = float(1.0 - min(1.0, mean_localization_error / max(EPS, float(point_match_radius))))
    count_agreement = _point_count_agreement(count_a, count_b)
    agreement_score = _weighted_mean([
        (f1, POINT_AGREEMENT_SCORE_WEIGHTS["f1"]),
        (localization_agreement, POINT_AGREEMENT_SCORE_WEIGHTS["localization"]),
        (count_agreement, POINT_AGREEMENT_SCORE_WEIGHTS["count_agreement"]),
    ])
    return PointAgreementMetrics(
        precision_at_radius=float(precision),
        recall_at_radius=float(recall),
        f1_at_radius=float(f1),
        mean_localization_error=float(mean_localization_error),
        localization_agreement=float(localization_agreement),
        count_agreement=float(count_agreement),
        matched_count=int(matched_count),
        true_positive_count=int(matched_count),
        false_positive_count=int(false_positive_count),
        false_negative_count=int(false_negative_count),
        point_count_a=int(count_a),
        point_count_b=int(count_b),
        agreement_score=float(agreement_score),
    )

def _prepare_mask_pairwise_descriptors(
    probabilities_by_model: dict[str, np.ndarray],
    masks_by_model: dict[str, np.ndarray],
    model_structures: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Precompute reusable per-model structures for symmetric mask agreement metrics."""

    descriptors: dict[str, dict[str, object]] = {}
    for model_id, probability in probabilities_by_model.items():
        prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
        clip_eps = np.float32(max(EPS, float(np.finfo(np.float32).eps)))
        prob_clipped = np.clip(prob, clip_eps, np.float32(1.0) - clip_eps)
        prob_sum = float(np.sum(prob, dtype=np.float64))
        prob_sq_sum = float(np.sum(np.square(prob, dtype=np.float32), dtype=np.float64))
        pixel_count = max(1.0, float(prob.size))
        mask = np.asarray(masks_by_model.get(model_id), dtype=bool)
        current_structure = (model_structures or {}).get(str(model_id)) or _mask_structure(mask, include_boundary_distance=True)
        boundary = np.asarray(current_structure.get('boundary'), dtype=bool) if current_structure.get('boundary') is not None else _boundary_mask(mask)
        dist_to_boundary = current_structure.get('boundary_dist')
        if dist_to_boundary is not None:
            dist_to_boundary = np.asarray(dist_to_boundary, dtype=np.float32)
        elif np.any(boundary) and (ndi is not None or cv2 is not None):
            dist_to_boundary = _distance_transform(~boundary)
        descriptors[str(model_id)] = {
            'prob': prob,
            'prob_sum': prob_sum,
            'prob_sq_sum': prob_sq_sum,
            'prob_mean': float(prob_sum / pixel_count),
            'prob_var': float(max(0.0, (prob_sq_sum / pixel_count) - (prob_sum / pixel_count) ** 2)),
            'log_prob': np.log(prob_clipped),
            'log_inv_prob': np.log1p(-prob_clipped),
            'mask': mask,
            'mask_area': int(np.count_nonzero(mask)),
            'boundary': boundary,
            'boundary_dist': dist_to_boundary,
            'centroid': _mask_centroid(mask),
            'structure': current_structure,
            'shape': tuple(int(v) for v in mask.shape),
        }
    return descriptors
