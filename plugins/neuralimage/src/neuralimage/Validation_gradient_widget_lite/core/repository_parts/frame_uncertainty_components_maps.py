"""repository: frame uncertainty components maps."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    MODEL_RISK_UNCERTAINTY_THRESHOLD,
    MODEL_RISK_WEIGHT_CLUSTER,
    MODEL_RISK_WEIGHT_FRACTION,
    MODEL_RISK_WEIGHT_MEAN,
    MODEL_RISK_WEIGHT_TOP,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.robust_normalize_display_map import (
    _largest_uncertain_region_fraction,
    _support_weights_from_probability,
    _top_weighted_uncertainty_mean,
    _uncertainty_map_from_probability,
)

def _frame_uncertainty_components_from_maps(
    uncertainty: np.ndarray,
    support_weights: np.ndarray,
    *,
    uncertainty_threshold: float,
    top_fraction: float,
    risk_weight_mean: float,
    risk_weight_fraction: float,
    risk_weight_top: float,
    risk_weight_cluster: float,
    sampled_uncertainty_values: np.ndarray | None = None,
    sampled_weight_values: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    support_weight_map = np.asarray(support_weights, dtype=np.float32)
    if support_weight_map.ndim != 2 or support_weight_map.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    uncertainty_map = np.asarray(uncertainty, dtype=np.float32)
    if uncertainty_map.shape != support_weight_map.shape:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    # The frame-level score must stay sensitive to structured local failures.
    # Using only mean(confidence) hides large bad pockets behind many easy pixels,
    # so the badness aggregation combines global uncertainty, the worst tail,
    # the low-confidence fraction, and the size of the largest bad component.
    valid_mask = np.isfinite(uncertainty_map)
    if support_mask is not None:
        candidate_mask = np.asarray(support_mask, dtype=bool)
        if candidate_mask.shape == uncertainty_map.shape and np.any(candidate_mask):
            valid_mask &= candidate_mask
    if not np.any(valid_mask):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    uncertainty_values = np.asarray(uncertainty_map[valid_mask], dtype=np.float32)
    if uncertainty_values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    unit_weights = np.ones_like(uncertainty_values, dtype=np.float32)
    uncertainty_cutoff = float(uncertainty_threshold)
    mean_uncertainty = float(np.mean(uncertainty_values, dtype=np.float64))
    low_conf_fraction = float(np.mean(uncertainty_values > uncertainty_cutoff, dtype=np.float64))
    top_uncertainty_mean = _top_weighted_uncertainty_mean(
        uncertainty_values,
        unit_weights,
        float(top_fraction),
        assume_valid=True,
    )
    largest_region_fraction = _largest_uncertain_region_fraction(
        uncertainty_map,
        valid_mask,
        uncertainty_threshold=uncertainty_cutoff,
    )
    # Larger score means a worse frame: global uncertainty captures overall drift,
    # the worst tail highlights severe local failures, the low-confidence fraction
    # measures spread, and the largest component penalizes structurally connected
    # bad regions that simple averaging would hide.
    denominator = max(EPS, float(risk_weight_mean + risk_weight_fraction + risk_weight_top + risk_weight_cluster))
    score = float(
        (
            float(risk_weight_mean) * mean_uncertainty
            + float(risk_weight_fraction) * low_conf_fraction
            + float(risk_weight_top) * top_uncertainty_mean
            + float(risk_weight_cluster) * largest_region_fraction
        )
        / denominator
    )
    return float(np.clip(score, 0.0, 1.0)), mean_uncertainty, low_conf_fraction, top_uncertainty_mean, largest_region_fraction

def _frame_uncertainty_components_from_probability(
    probability: np.ndarray,
    *,
    support_threshold: float,
    uncertainty_threshold: float = MODEL_RISK_UNCERTAINTY_THRESHOLD,
    top_fraction: float = MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    risk_weight_mean: float = MODEL_RISK_WEIGHT_MEAN,
    risk_weight_fraction: float = MODEL_RISK_WEIGHT_FRACTION,
    risk_weight_top: float = MODEL_RISK_WEIGHT_TOP,
    risk_weight_cluster: float = MODEL_RISK_WEIGHT_CLUSTER,
) -> tuple[float, float, float, float, float]:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    support_weights = _support_weights_from_probability(prob, support_threshold)
    if prob.ndim != 2 or prob.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    uncertainty = _uncertainty_map_from_probability(prob)
    return _frame_uncertainty_components_from_maps(
        uncertainty,
        support_weights,
        uncertainty_threshold=float(uncertainty_threshold),
        top_fraction=float(top_fraction),
        risk_weight_mean=float(risk_weight_mean),
        risk_weight_fraction=float(risk_weight_fraction),
        risk_weight_top=float(risk_weight_top),
        risk_weight_cluster=float(risk_weight_cluster),
    )
