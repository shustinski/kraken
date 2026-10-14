"""repository: aggregate inter model point."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    INTER_MODEL_POINT_SCORE_WEIGHTS,
    INTER_MODEL_POLYGON_SCORE_WEIGHTS,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.match_point_sets import _polygon_bce_score

def _aggregate_inter_model_polygon_scores(pairwise_rows: tuple[dict[str, object], ...]) -> dict[str, float]:
    """Aggregate pairwise polygon comparison metrics into one frame-level score bundle."""

    if not pairwise_rows:
        return {}
    iou = float(np.mean(np.asarray([float(row.get("iou", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    dice = float(np.mean(np.asarray([float(row.get("dice", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    bce = float(np.mean(np.asarray([float(row.get("bce", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    iou_score = float(max(0.0, min(iou, 1.0)) * 100.0)
    dice_score = float(max(0.0, min(dice, 1.0)) * 100.0)
    bce_score = _polygon_bce_score(bce)
    weight_sum = float(sum(INTER_MODEL_POLYGON_SCORE_WEIGHTS.values()))
    overall = (
        iou_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["iou"]) +
        dice_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["dice"]) +
        bce_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["bce"])
    ) / max(EPS, weight_sum)
    return {
        "iou": iou,
        "dice": dice,
        "bce": bce,
        "iou_score": iou_score,
        "dice_score": dice_score,
        "polygon_bce_score": bce_score,
        "overall_polygon_score": float(overall),
    }

def _aggregate_inter_model_point_scores(pairwise_rows: tuple[dict[str, object], ...], point_match_radius: float) -> dict[str, float]:
    """Aggregate pairwise point comparison metrics into one frame-level score bundle."""

    if not pairwise_rows:
        return {}
    precision = float(np.mean(np.asarray([float(row.get("precision", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    recall = float(np.mean(np.asarray([float(row.get("recall", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    f1 = float(np.mean(np.asarray([float(row.get("f1", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    mean_localization_distance = float(np.mean(np.asarray([float(row.get("mean_localization_error", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    tp = float(np.mean(np.asarray([float(row.get("matched_count", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    fp = float(np.mean(np.asarray([max(0.0, float(row.get("point_count_a", 0.0)) - float(row.get("matched_count", 0.0))) for row in pairwise_rows], dtype=np.float64)))
    fn = float(np.mean(np.asarray([max(0.0, float(row.get("point_count_b", 0.0)) - float(row.get("matched_count", 0.0))) for row in pairwise_rows], dtype=np.float64)))
    precision_score = float(max(0.0, min(precision, 1.0)) * 100.0)
    recall_score = float(max(0.0, min(recall, 1.0)) * 100.0)
    f1_score = float(max(0.0, min(f1, 1.0)) * 100.0)
    localization_score = float(max(0.0, 100.0 * (1.0 - min(mean_localization_distance / max(EPS, float(point_match_radius)), 1.0))))
    weight_sum = float(sum(INTER_MODEL_POINT_SCORE_WEIGHTS.values()))
    overall = (
        precision_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["precision"]) +
        recall_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["recall"]) +
        f1_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["f1"]) +
        localization_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["localization"])
    ) / max(EPS, weight_sum)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_localization_distance": mean_localization_distance,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "localization_score": localization_score,
        "overall_point_score": float(overall),
    }
