"""repository: pairwise mask metrics."""
from __future__ import annotations
import math
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MASK_AGREEMENT_SCORE_WEIGHTS,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import (
    _clip01,
    _weighted_mean,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import (
    _distance_similarity,
    _frame_diagonal,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.match_point_sets import _mask_count_agreement

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.structural_penalties import _hausdorff_distance

def _pairwise_mask_metrics(
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, float]:
    """Compute all symmetric mask agreement metrics from precomputed per-model descriptors."""

    first_prob = np.asarray(first['prob'], dtype=np.float32)
    second_prob = np.asarray(second['prob'], dtype=np.float32)
    first_mask = np.asarray(first['mask'], dtype=bool)
    second_mask = np.asarray(second['mask'], dtype=bool)
    shape = tuple(int(v) for v in first['shape'])

    intersection_mask = int(np.count_nonzero(first_mask & second_mask))
    area_first = int(first['mask_area'])
    area_second = int(second['mask_area'])
    union_mask = int(area_first + area_second - intersection_mask)

    prob_intersection = float(np.sum(first_prob * second_prob, dtype=np.float64))
    prob_sum_first = float(first['prob_sum'])
    prob_sum_second = float(second['prob_sum'])
    prob_union = float(prob_sum_first + prob_sum_second - prob_intersection)
    pixel_count = max(1.0, float(first_prob.size))

    soft_dice = 1.0 if (prob_sum_first + prob_sum_second) <= EPS else float((2.0 * prob_intersection + EPS) / (prob_sum_first + prob_sum_second + EPS))
    soft_iou = 1.0 if prob_union <= EPS else float((prob_intersection + EPS) / (prob_union + EPS))

    mu_first = float(first.get('prob_mean', 0.0))
    mu_second = float(second.get('prob_mean', 0.0))
    sigma_first = float(max(0.0, float(first.get('prob_var', 0.0))))
    sigma_second = float(max(0.0, float(second.get('prob_var', 0.0))))
    sigma_cross = float((prob_intersection / pixel_count) - (mu_first * mu_second))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    denominator = (mu_first ** 2 + mu_second ** 2 + c1) * (sigma_first + sigma_second + c2)
    ssim = 1.0 if denominator <= EPS and np.allclose(first_prob, second_prob) else (0.0 if denominator <= EPS else float(_clip01(((2.0 * mu_first * mu_second + c1) * (2.0 * sigma_cross + c2)) / denominator)))

    dice = 1.0 if (area_first + area_second) == 0 else float((2.0 * intersection_mask + EPS) / (area_first + area_second + EPS))
    iou = 1.0 if union_mask == 0 else float((intersection_mask + EPS) / (union_mask + EPS))

    diagonal = _frame_diagonal(shape)
    first_center = first['centroid']
    second_center = second['centroid']
    if first_center is None and second_center is None:
        centroid_distance = 0.0
    elif first_center is None or second_center is None:
        centroid_distance = diagonal
    else:
        centroid_distance = float(math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))
    centroid_similarity = _distance_similarity(centroid_distance, shape)

    first_boundary = np.asarray(first['boundary'], dtype=bool)
    second_boundary = np.asarray(second['boundary'], dtype=bool)
    if not np.any(first_boundary) and not np.any(second_boundary):
        hausdorff_distance = 0.0
    elif not np.any(first_boundary) or not np.any(second_boundary):
        hausdorff_distance = diagonal
    elif first.get('boundary_dist') is not None and second.get('boundary_dist') is not None:
        directed_first = float(np.max(np.asarray(second['boundary_dist'], dtype=np.float32)[first_boundary])) if np.any(first_boundary) else 0.0
        directed_second = float(np.max(np.asarray(first['boundary_dist'], dtype=np.float32)[second_boundary])) if np.any(second_boundary) else 0.0
        hausdorff_distance = float(max(directed_first, directed_second))
    else:
        hausdorff_distance = _hausdorff_distance(first_mask, second_mask)
    hausdorff_similarity = _distance_similarity(hausdorff_distance, shape)

    diff = first_prob - second_prob
    mae = float(np.mean(np.abs(diff), dtype=np.float64)) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(np.square(diff, dtype=np.float32), dtype=np.float64))) if diff.size else 0.0
    first_log = np.asarray(first.get('log_prob'), dtype=np.float32)
    first_log_inv = np.asarray(first.get('log_inv_prob'), dtype=np.float32)
    second_log = np.asarray(second.get('log_prob'), dtype=np.float32)
    second_log_inv = np.asarray(second.get('log_inv_prob'), dtype=np.float32)
    forward = -(first_prob * second_log + (1.0 - first_prob) * second_log_inv)
    backward = -(second_prob * first_log + (1.0 - second_prob) * first_log_inv)
    bce = float(np.mean((forward + backward) * 0.5, dtype=np.float64)) if forward.size else 0.0

    count_agreement = _mask_count_agreement(
        int(np.asarray(first['structure']['component_count']).item() if hasattr(first['structure']['component_count'], 'item') else first['structure']['component_count']),
        int(np.asarray(second['structure']['component_count']).item() if hasattr(second['structure']['component_count'], 'item') else second['structure']['component_count']),
    )
    agreement_score = _weighted_mean([
        (float(soft_dice), MASK_AGREEMENT_SCORE_WEIGHTS['soft_dice']),
        (float(soft_iou), MASK_AGREEMENT_SCORE_WEIGHTS['soft_iou']),
        (float(ssim), MASK_AGREEMENT_SCORE_WEIGHTS['ssim']),
        (float(dice), MASK_AGREEMENT_SCORE_WEIGHTS['dice']),
        (float(iou), MASK_AGREEMENT_SCORE_WEIGHTS['iou']),
        (float(hausdorff_similarity), MASK_AGREEMENT_SCORE_WEIGHTS['hausdorff_term']),
        (float(centroid_similarity), MASK_AGREEMENT_SCORE_WEIGHTS['centroid_term']),
    ])

    return {
        'soft_dice': float(soft_dice),
        'soft_iou': float(soft_iou),
        'ssim': float(ssim),
        'dice': float(dice),
        'iou': float(iou),
        'hausdorff_distance': float(hausdorff_distance),
        'hausdorff_similarity': float(hausdorff_similarity),
        'centroid_distance': float(centroid_distance),
        'centroid_similarity': float(centroid_similarity),
        'mae': float(mae),
        'rmse': float(rmse),
        'bce': float(bce),
        'count_agreement': float(count_agreement),
        'agreement_score': float(agreement_score),
    }
