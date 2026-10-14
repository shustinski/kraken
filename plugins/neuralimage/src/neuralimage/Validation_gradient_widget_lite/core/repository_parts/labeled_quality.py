"""repository: labeled quality."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MASK_SUPERVISED_SCORE_WEIGHTS,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    LabeledModelMetrics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import (
    _clip01,
    _weighted_mean,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import (
    _boundary_f1,
    _centroid_distance,
    _dice,
    _distance_similarity,
    _iou,
    _mae,
    _mask_structure,
    _precision,
    _recall,
    _rmse,
    _soft_dice,
    _soft_iou,
    _ssim_score,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.structural_penalties import (
    _hausdorff_distance,
    _structural_penalties,
)

def _labeled_quality(
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    pred_structure: dict[str, object] | None = None,
    gt_structure: dict[str, object] | None = None,
    boundary_radius: int = 1,
) -> LabeledModelMetrics:
    current_pred_structure = pred_structure or _mask_structure(pred_mask, include_boundary_distance=True)
    current_gt_structure = gt_structure or _mask_structure(gt_mask, include_boundary_distance=True)
    gt_float = np.asarray(gt_mask, dtype=np.float32)
    pred_float = np.clip(np.asarray(pred_prob, dtype=np.float32), 0.0, 1.0)

    soft_dice = _soft_dice(pred_float, gt_float)
    soft_iou = _soft_iou(pred_float, gt_float)
    ssim = _ssim_score(pred_float, gt_float)
    dice = _dice(pred_mask, gt_mask)
    iou = _iou(pred_mask, gt_mask)
    precision = _precision(pred_mask, gt_mask)
    recall = _recall(pred_mask, gt_mask)
    boundary_f1 = _boundary_f1(pred_mask, gt_mask, radius=max(1, int(boundary_radius)))
    hausdorff_distance = _hausdorff_distance(
        pred_mask,
        gt_mask,
        first_structure=current_pred_structure,
        second_structure=current_gt_structure,
    )
    centroid_distance = _centroid_distance(pred_mask, gt_mask)
    frame_shape = tuple(int(v) for v in np.asarray(pred_mask).shape)
    hausdorff_similarity = _distance_similarity(hausdorff_distance, frame_shape)
    centroid_similarity = _distance_similarity(centroid_distance, frame_shape)
    mae = _mae(pred_float, gt_float)
    rmse = _rmse(pred_float, gt_float)

    delta_cc, break_count, false_bridge_count, skeleton_delta = _structural_penalties(
        pred_mask,
        gt_mask,
        pred_structure=current_pred_structure,
        gt_structure=current_gt_structure,
    )
    pred_count = int(current_pred_structure["component_count"])
    gt_count = int(current_gt_structure["component_count"])
    count_error = float(_clip01(abs(pred_count - gt_count) / max(1.0, float(gt_count))))
    connected_component_error = float(_clip01(abs(pred_count - gt_count) / max(1.0, float(gt_count))))
    quality = _weighted_mean([
        (float(soft_dice), MASK_SUPERVISED_SCORE_WEIGHTS["soft_dice"]),
        (float(soft_iou), MASK_SUPERVISED_SCORE_WEIGHTS["soft_iou"]),
        (float(ssim), MASK_SUPERVISED_SCORE_WEIGHTS["ssim"]),
        (float(dice), MASK_SUPERVISED_SCORE_WEIGHTS["dice"]),
        (float(iou), MASK_SUPERVISED_SCORE_WEIGHTS["iou"]),
        (float(hausdorff_similarity), MASK_SUPERVISED_SCORE_WEIGHTS["hausdorff_term"]),
        (float(centroid_similarity), MASK_SUPERVISED_SCORE_WEIGHTS["centroid_term"]),
    ])
    error = float(1.0 - quality)
    return LabeledModelMetrics(
        soft_dice=float(soft_dice),
        soft_iou=float(soft_iou),
        ssim=float(ssim),
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        count_error=float(count_error),
        connected_component_error=float(connected_component_error),
        hausdorff_distance=float(hausdorff_distance),
        hausdorff_similarity=float(hausdorff_similarity),
        centroid_distance=float(centroid_distance),
        centroid_similarity=float(centroid_similarity),
        mae=float(mae),
        rmse=float(rmse),
        boundary_f1=float(boundary_f1),
        delta_connected_components=float(delta_cc),
        break_count=int(break_count),
        false_bridge_count=int(false_bridge_count),
        skeleton_length_delta=float(skeleton_delta),
        quality_score=float(quality),
        error_score=float(error),
    )
