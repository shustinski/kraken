"""repository: frame uncertainty components points."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    MODEL_RISK_UNCERTAINTY_THRESHOLD,
    MODEL_RISK_WEIGHT_CLUSTER,
    MODEL_RISK_WEIGHT_FRACTION,
    MODEL_RISK_WEIGHT_MEAN,
    MODEL_RISK_WEIGHT_TOP,
    POINT_SUPPORT_THRESHOLD,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.point_uncertainty_cluster_fraction import _point_uncertainty_cluster_fraction

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.robust_normalize_display_map import _top_weighted_uncertainty_mean

def _frame_uncertainty_components_from_points(
    point_probabilities: np.ndarray,
    point_coordinates: tuple[tuple[float, float, float], ...],
    *,
    support_threshold: float = POINT_SUPPORT_THRESHOLD,
    uncertainty_threshold: float = MODEL_RISK_UNCERTAINTY_THRESHOLD,
    top_fraction: float = MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    risk_weight_mean: float = MODEL_RISK_WEIGHT_MEAN,
    risk_weight_fraction: float = MODEL_RISK_WEIGHT_FRACTION,
    risk_weight_top: float = MODEL_RISK_WEIGHT_TOP,
    risk_weight_cluster: float = MODEL_RISK_WEIGHT_CLUSTER,
) -> tuple[float, float, float, float, float]:
    probabilities = np.clip(np.asarray(point_probabilities, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if probabilities.size == 0 or not point_coordinates:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    support_weights = np.zeros_like(probabilities, dtype=np.float32)
    support_mask = probabilities >= float(support_threshold)
    if np.any(support_mask):
        support_weights[support_mask] = (probabilities[support_mask] - float(support_threshold)) / max(EPS, 1.0 - float(support_threshold))
    support_weights = np.clip(support_weights, 0.0, 1.0).astype(np.float32)
    if not np.any(support_weights > 0.0):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    uncertainty = 1.0 - np.clip(2.0 * np.abs(probabilities - 0.5), 0.0, 1.0)
    positive_mask = np.isfinite(uncertainty)
    uncertainty_values = np.asarray(uncertainty[positive_mask], dtype=np.float64)
    if uncertainty_values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean_uncertainty = float(np.mean(uncertainty_values, dtype=np.float64))
    low_conf_fraction = float(np.mean(uncertainty_values > float(uncertainty_threshold), dtype=np.float64))
    top_uncertainty_mean = _top_weighted_uncertainty_mean(
        uncertainty_values,
        np.ones_like(uncertainty_values, dtype=np.float64),
        float(top_fraction),
        assume_valid=True,
    )
    largest_region_fraction = _point_uncertainty_cluster_fraction(
        point_coordinates,
        uncertainty,
        np.asarray(positive_mask, dtype=bool),
        uncertainty_threshold=float(uncertainty_threshold),
    )
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
