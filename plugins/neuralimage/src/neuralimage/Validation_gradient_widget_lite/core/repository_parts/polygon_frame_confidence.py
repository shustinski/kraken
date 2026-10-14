"""repository: polygon frame confidence."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    MODEL_RISK_UNCERTAINTY_THRESHOLD,
    MODEL_RISK_WEIGHT_CLUSTER,
    MODEL_RISK_WEIGHT_FRACTION,
    MODEL_RISK_WEIGHT_MEAN,
    MODEL_RISK_WEIGHT_TOP,
    POLYGON_SUPPORT_THRESHOLD,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidenceDebugData,
    PolygonConfidenceMetrics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.frame_uncertainty_components_maps import _frame_uncertainty_components_from_maps

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.internal_confidence_probability_map import _internal_confidence_probability_map

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _boundary_mask,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.robust_normalize_display_map import _support_weights_from_probability

def _polygon_frame_confidence(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    uncertainty_delta: float = MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    summary_metric: str = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    include_debug: bool = False,
    allow_binary_proxy: bool = True,
) -> PolygonConfidenceMetrics:
    """Compute lightweight frame-level polygon confidence without object-aware geometry refinement."""

    strong = np.asarray(mask, dtype=bool)
    prob = _internal_confidence_probability_map(
        probability,
        support_mask=strong,
        allow_binary_proxy=allow_binary_proxy,
    )
    if strong.shape != prob.shape:
        strong = np.asarray(strong, dtype=bool)
        if strong.shape != prob.shape:
            strong = np.zeros_like(prob, dtype=bool)
    prob_distance = np.abs(prob - np.float32(0.5)).astype(np.float32, copy=False)
    confidence = np.clip(2.0 * prob_distance, 0.0, 1.0).astype(np.float32, copy=False)
    uncertainty = (1.0 - confidence).astype(np.float32, copy=False)
    support_weights = _support_weights_from_probability(prob, POLYGON_SUPPORT_THRESHOLD)
    support_mask = support_weights > 0.0
    support_probability_values = np.asarray(prob[support_mask], dtype=np.float32)
    support_weight_values = np.asarray(support_weights[support_mask], dtype=np.float32)
    support_uncertainty_values = np.asarray(uncertainty[support_mask], dtype=np.float32)
    uncertainty_band = float(max(0.0, uncertainty_delta))
    if np.any(support_mask):
        support_weight_sum = float(np.sum(support_weight_values, dtype=np.float64))
        support_uncertain_mask = np.asarray(
            np.abs(support_probability_values - np.float32(0.5)) <= uncertainty_band,
            dtype=np.float32,
        )
        uncertain_fraction = float(
            np.sum(
                support_weight_values * support_uncertain_mask,
                dtype=np.float64,
            )
            / max(EPS, support_weight_sum)
        )
    else:
        uncertain_fraction = 0.0
    boundary_mask = _boundary_mask(strong) if np.any(strong) else _boundary_mask(support_mask)
    boundary_uncertainty = float(np.mean(uncertainty[boundary_mask], dtype=np.float64)) if np.any(boundary_mask) else 0.0
    core_threshold = max(0.65, float(np.quantile(support_probability_values, 0.65)) if support_probability_values.size > 0 else 0.65)
    core_mask = support_mask & (prob >= core_threshold)
    core_confidence = float(np.mean(confidence[core_mask], dtype=np.float64)) if np.any(core_mask) else 0.0
    labels, polygon_count = _label_components(strong if np.any(strong) else support_mask)
    weight_total = float(np.sum(support_weight_values, dtype=np.float64))
    mean_confidence = float(
        np.sum(support_weight_values * np.asarray(confidence[support_mask], dtype=np.float32), dtype=np.float64) / max(EPS, weight_total)
    ) if weight_total > 0.0 else 0.0
    mean_probability = float(
        np.sum(support_weight_values * support_probability_values, dtype=np.float64) / max(EPS, weight_total)
    ) if weight_total > 0.0 else 0.0
    frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = _frame_uncertainty_components_from_maps(
        uncertainty,
        support_weights,
        uncertainty_threshold=float(MODEL_RISK_UNCERTAINTY_THRESHOLD),
        top_fraction=float(MODEL_RISK_TOP_UNCERTAIN_FRACTION),
        risk_weight_mean=float(MODEL_RISK_WEIGHT_MEAN),
        risk_weight_fraction=float(MODEL_RISK_WEIGHT_FRACTION),
        risk_weight_top=float(MODEL_RISK_WEIGHT_TOP),
        risk_weight_cluster=float(MODEL_RISK_WEIGHT_CLUSTER),
        sampled_uncertainty_values=support_uncertainty_values,
        sampled_weight_values=support_weight_values,
        support_mask=support_mask,
    )
    area_fraction = float(np.count_nonzero(support_mask) / max(1, support_mask.size))
    debug_data = None
    if include_debug:
        debug_data = PolygonConfidenceDebugData(
            preprocessed_probability=np.asarray(prob, dtype=np.float32),
            low_mask=np.asarray(support_mask, dtype=bool),
            high_mask=np.asarray(strong, dtype=bool),
            merged_mask=np.asarray(strong if np.any(strong) else support_mask, dtype=bool),
            object_labels=np.asarray(labels, dtype=np.int32),
            candidate_rows=(),
            timings_ms={},
        )
    return PolygonConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        mean_uncertainty=mean_uncertainty,
        uncertain_support_fraction=low_conf_fraction,
        top_uncertainty_mean=worst_tail_uncertainty,
        largest_uncertain_region_fraction=largest_low_conf_component,
        mean_object_confidence=mean_confidence,
        mean_core_confidence=core_confidence,
        mean_boundary_uncertainty=boundary_uncertainty,
        mean_weighted_confidence=mean_confidence,
        mean_object_probability=mean_probability,
        uncertain_fraction=uncertain_fraction,
        mean_transition_width=0.0,
        object_area_fraction=area_fraction,
        polygon_count=int(polygon_count),
        summary_metric=str(summary_metric),
        low_conf_fraction=low_conf_fraction,
        worst_tail_uncertainty=worst_tail_uncertainty,
        largest_low_conf_component=largest_low_conf_component,
        objects=tuple(),
        debug_data=debug_data,
    )
