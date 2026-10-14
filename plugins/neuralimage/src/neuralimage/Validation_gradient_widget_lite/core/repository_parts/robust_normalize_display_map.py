"""repository: robust normalize display map."""
from __future__ import annotations
import math
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    POLYGON_SUPPORT_THRESHOLD,
)
from neuralimage.Validation_gradient_widget_lite.core.confidence_maps import build_model_uncertainty

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _label_components

def _robust_normalize_display_map(
    values: np.ndarray,
    *,
    focus_mask: np.ndarray | None = None,
    lower_percentile: float = 10.0,
    upper_percentile: float = 99.0,
    gamma: float = 0.75,
    invert: bool = False,
) -> np.ndarray:
    """Stretch a narrow value range so close confidence levels remain visible in the UI."""

    array = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if array.ndim != 2 or array.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    finite_mask = np.isfinite(array)
    if focus_mask is not None:
        mask = np.asarray(focus_mask, dtype=bool)
        if mask.shape == array.shape:
            finite_mask &= mask
    focus_values = np.asarray(array[finite_mask], dtype=np.float32)
    if focus_values.size == 0:
        focus_values = np.asarray(array[np.isfinite(array)], dtype=np.float32)
    if focus_values.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    low = float(np.percentile(focus_values, float(lower_percentile)))
    high = float(np.percentile(focus_values, float(upper_percentile)))
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) <= EPS:
        low = float(np.min(focus_values))
        high = float(np.max(focus_values))
    if abs(high - low) <= EPS:
        median = float(np.median(focus_values))
        spread = float(np.max(np.abs(focus_values - median))) if focus_values.size else 0.0
        normalized = 0.5 + 0.5 * (array - median) / max(EPS, spread)
    else:
        normalized = (array - low) / max(EPS, high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    gamma_value = float(max(0.05, gamma))
    if abs(gamma_value - 1.0) > EPS:
        normalized = np.power(normalized, gamma_value).astype(np.float32, copy=False)
    if invert:
        normalized = 1.0 - normalized
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

def _support_weights_from_probability(probability: np.ndarray, support_threshold: float) -> np.ndarray:
    """Map probabilities into support weights used by confidence overlays and scores."""

    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    threshold = float(max(0.0, min(0.999, support_threshold)))
    weights = np.zeros_like(prob, dtype=np.float32)
    if prob.size == 0:
        return weights
    support_mask = prob >= threshold
    if not np.any(support_mask):
        return weights
    weights[support_mask] = (prob[support_mask] - threshold) / max(EPS, 1.0 - threshold)
    return np.clip(weights, 0.0, 1.0).astype(np.float32)

def _uncertainty_map_from_probability(probability: np.ndarray) -> np.ndarray:
    return build_model_uncertainty(probability)

def _confidence_display_map_from_probability(
    probability: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    support_threshold: float = POLYGON_SUPPORT_THRESHOLD,
) -> np.ndarray:
    """Build a display-only confidence map with robust contrast enhancement.

    The raw confidence metric remains unchanged. The UI map is normalized separately
    so dense high-confidence ranges (for example 0.80-0.95 probabilities) remain
    visually separable and low-confidence pockets become easier to spot.
    """

    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    if prob.ndim != 2 or prob.size == 0:
        return np.zeros_like(prob, dtype=np.float32)
    focus_mask = None
    if support_mask is not None:
        candidate_mask = np.asarray(support_mask, dtype=bool)
        if candidate_mask.shape == prob.shape and np.any(candidate_mask):
            focus_mask = candidate_mask
    if focus_mask is None:
        support_weights = _support_weights_from_probability(prob, support_threshold)
        candidate_mask = np.asarray(support_weights > 0.0, dtype=bool)
        if np.any(candidate_mask):
            focus_mask = candidate_mask
    uncertainty = _uncertainty_map_from_probability(prob)
    enhanced_uncertainty = _robust_normalize_display_map(
        uncertainty,
        focus_mask=focus_mask,
        lower_percentile=10.0,
        upper_percentile=99.0,
        gamma=0.75,
        invert=False,
    )
    return np.clip(1.0 - enhanced_uncertainty, 0.0, 1.0).astype(np.float32, copy=False)

def _top_weighted_uncertainty_mean(
    uncertainty_values: np.ndarray,
    weight_values: np.ndarray,
    top_fraction: float,
    *,
    assume_valid: bool = False,
) -> float:
    uncertainties = np.asarray(uncertainty_values, dtype=np.float32).reshape(-1)
    weights = np.asarray(weight_values, dtype=np.float32).reshape(-1)
    if uncertainties.size == 0 or weights.size == 0:
        return 0.0
    if not assume_valid:
        valid_mask = np.isfinite(uncertainties) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid_mask):
            return 0.0
        uncertainties = uncertainties[valid_mask]
        weights = weights[valid_mask]
    count = max(1, int(math.ceil(float(uncertainties.size) * max(0.0, float(top_fraction)))))
    count = min(count, int(uncertainties.size))
    if count <= 0:
        return 0.0
    if count >= int(uncertainties.size):
        return float(np.sum(uncertainties * weights, dtype=np.float64) / max(EPS, float(np.sum(weights, dtype=np.float64))))
    top_indices = np.argpartition(-uncertainties, count - 1)[:count]
    top_uncertainty = uncertainties[top_indices]
    top_weights = weights[top_indices]
    return float(np.sum(top_uncertainty * top_weights, dtype=np.float64) / max(EPS, float(np.sum(top_weights, dtype=np.float64))))

def _largest_uncertain_region_fraction(
    uncertainty: np.ndarray,
    support_mask: np.ndarray,
    *,
    uncertainty_threshold: float,
) -> float:
    support = np.asarray(support_mask, dtype=bool)
    if support.size == 0 or not np.any(support):
        return 0.0
    uncertainty_map = np.asarray(uncertainty, dtype=np.float32)
    uncertain_support = support & np.isfinite(uncertainty_map) & (uncertainty_map > float(uncertainty_threshold))
    if not np.any(uncertain_support):
        return 0.0
    labels, component_count = _label_components(uncertain_support)
    if component_count <= 0:
        return 0.0
    component_sizes = np.bincount(np.asarray(labels, dtype=np.int32).reshape(-1))
    largest_area = int(np.max(component_sizes[1:], initial=0)) if component_sizes.size > 1 else 0
    # Normalize by the full frame area so equally large low-confidence regions stay
    # comparable across frames, even when their confident support area differs.
    return float(largest_area / max(1, support.size))
