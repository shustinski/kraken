"""Frame, polygon, and point confidence calculations."""

from __future__ import annotations

from .mask_primitives import (
    _boundary_mask,
    _distance_transform,
    _label_components,
)

from .repository_shared import (
    EPS,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    MODEL_RISK_UNCERTAINTY_THRESHOLD,
    MODEL_RISK_WEIGHT_CLUSTER,
    MODEL_RISK_WEIGHT_FRACTION,
    MODEL_RISK_WEIGHT_MEAN,
    MODEL_RISK_WEIGHT_TOP,
    ModelOutputConfidenceMetrics,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POINT_SUPPORT_THRESHOLD,
    POLYGON_CONFIDENCE_ADAPTIVE_HIGH_OFFSET,
    POLYGON_CONFIDENCE_ADAPTIVE_LOW_OFFSET,
    POLYGON_CONFIDENCE_ADAPTIVE_RADIUS,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_ASPECT,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_DROP,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_RETAINED_FRACTION,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_PROFILE_QUANTILE,
    POLYGON_CONFIDENCE_DOMINANT_LARGE_AREA,
    POLYGON_CONFIDENCE_DOMINANT_LOCK_RADIUS,
    POLYGON_CONFIDENCE_DOMINANT_MIN_AREA,
    POLYGON_CONFIDENCE_DOMINANT_MIN_ASPECT_RATIO,
    POLYGON_CONFIDENCE_DOMINANT_MIN_EXTENT,
    POLYGON_CONFIDENCE_DOMINANT_MIN_MEAN_PROBABILITY,
    POLYGON_CONFIDENCE_ELONGATED_HORIZONTAL_RADIUS,
    POLYGON_CONFIDENCE_ELONGATED_MIN_AREA,
    POLYGON_CONFIDENCE_ELONGATED_MIN_ASPECT_RATIO,
    POLYGON_CONFIDENCE_ELONGATED_VERTICAL_RADIUS,
    POLYGON_CONFIDENCE_ENABLE_WATERSHED,
    POLYGON_CONFIDENCE_HOLE_MIN_AREA,
    POLYGON_CONFIDENCE_HOLE_PROBABILITY_MAX,
    POLYGON_CONFIDENCE_HOLE_PROBABILITY_SCALE,
    POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    POLYGON_CONFIDENCE_HYSTERESIS_LOW_RATIO,
    POLYGON_CONFIDENCE_LARGE_POLYGON_BAND_EXPAND,
    POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_COVERAGE_MIN,
    POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_DELTA,
    POLYGON_CONFIDENCE_LARGE_POLYGON_LOW_SCALE,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MAJOR_CLOSE_RADIUS,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MINOR_CLOSE_RADIUS,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_AREA,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_ASPECT_RATIO,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_EXTENT,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_MAJOR_SPAN,
    POLYGON_CONFIDENCE_LARGE_POLYGON_ROI_PADDING,
    POLYGON_CONFIDENCE_LARGE_POLYGON_SEED_LOW_SCALE,
    POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_RADIUS,
    POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_STRENGTH,
    POLYGON_CONFIDENCE_MERGE_DISTANCE,
    POLYGON_CONFIDENCE_MERGE_IOU_THRESHOLD,
    POLYGON_CONFIDENCE_PREPROC_GAUSSIAN_SIGMA,
    POLYGON_CONFIDENCE_PREPROC_MEDIAN_RADIUS,
    POLYGON_CONFIDENCE_PROPOSAL_MEAN_FLOOR,
    POLYGON_CONFIDENCE_PROPOSAL_MIN_AREA,
    POLYGON_CONFIDENCE_PROPOSAL_PEAK_FLOOR,
    POLYGON_CONFIDENCE_SEPARATION_BARRIER_DILATE_RADIUS,
    POLYGON_CONFIDENCE_SEPARATION_BARRIER_THRESHOLD,
    POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_CONTRAST_WEIGHT,
    POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_LOW_WEIGHT,
    POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_UNCERTAINTY_WEIGHT,
    POLYGON_CONFIDENCE_SEPARATION_BRIDGE_BARRIER_THRESHOLD,
    POLYGON_CONFIDENCE_SEPARATION_BRIDGE_PROBABILITY_MAX,
    POLYGON_CONFIDENCE_SEPARATION_CORE_MIN_AREA,
    POLYGON_CONFIDENCE_SEPARATION_ROI_PADDING,
    POLYGON_CONFIDENCE_SMALL_HIGH_SCALE,
    POLYGON_CONFIDENCE_SMALL_LOW_SCALE,
    POLYGON_CONFIDENCE_SMALL_MAX_AREA,
    POLYGON_CONFIDENCE_SPILL_BORDER_COVERAGE_MIN,
    POLYGON_CONFIDENCE_SPILL_BOUNDARY_SEPARATION_MAX,
    POLYGON_CONFIDENCE_SPILL_CROSS_AXIS_MAX,
    POLYGON_CONFIDENCE_SPILL_LARGE_AREA_FRACTION,
    POLYGON_CONFIDENCE_SPILL_LARGE_EXTENT,
    POLYGON_CONFIDENCE_SPILL_LOW_TEXTURE_MAX,
    POLYGON_CONFIDENCE_SPILL_MEAN_PROBABILITY_MAX,
    POLYGON_CONFIDENCE_SPILL_PEAK_MARGIN_MAX,
    POLYGON_CONFIDENCE_SPILL_PROMINENCE_MIN,
    POLYGON_CONFIDENCE_SPILL_RIBBON_ASPECT_MIN,
    POLYGON_CONFIDENCE_SPILL_STRONG_AREA_FRACTION_MIN,
    POLYGON_CONFIDENCE_SPILL_STRONG_AXIS_COVERAGE_MIN,
    POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    POLYGON_CONFIDENCE_VALLEY_MINOR_COVERAGE_MIN,
    POLYGON_CONFIDENCE_WATERSHED_SEED_MIN_AREA,
    POLYGON_SUPPORT_THRESHOLD,
    PointConfidenceMetrics,
    PointObjectConfidence,
    PolygonConfidenceDebugData,
    PolygonConfidenceMetrics,
    PolygonConfidencePipelineConfig,
    _is_binary_like_probability,
    build_model_uncertainty,
    math,
    normalize_algorithmic_confidence,
    np,
)


def _mask_from_gray(gray: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    value = int(round(max(0.0, min(1.0, threshold)) * 255.0))
    return np.asarray(gray >= value, dtype=bool)


def _prob_from_gray(gray: np.ndarray) -> np.ndarray:
    return np.asarray(gray, dtype=np.float32) / 255.0


def _internal_confidence_probability_map(
    probability: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    allow_binary_proxy: bool = True,
) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    mask_bool = (
        np.asarray(support_mask, dtype=bool) if support_mask is not None else np.asarray(prob >= 0.5, dtype=bool)
    )
    if prob.size == 0 or not allow_binary_proxy or not _is_binary_like_probability(prob):
        return prob
    if mask_bool.shape != prob.shape:
        mask_bool = np.asarray(prob >= 0.5, dtype=bool)
    if not np.any(mask_bool):
        return prob
    inside = _distance_transform(mask_bool)
    outside = _distance_transform(~mask_bool)
    signed_distance = np.asarray(inside - outside, dtype=np.float32)
    scale = float(np.percentile(np.abs(signed_distance), 95.0))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.max(np.abs(signed_distance))) if signed_distance.size else 1.0
    scale = max(scale, 1.0)
    proxy = np.clip(0.5 + 0.5 * signed_distance / scale, 0.0, 1.0).astype(np.float32)
    proxy[mask_bool] = np.maximum(proxy[mask_bool], 0.5 + 0.5 * np.clip(inside[mask_bool] / scale, 0.0, 1.0))
    proxy[~mask_bool] = np.minimum(proxy[~mask_bool], 0.5 - 0.5 * np.clip(outside[~mask_bool] / scale, 0.0, 1.0))
    return np.clip(proxy, 0.0, 1.0).astype(np.float32)


def _confidence_map_from_probability(probability: np.ndarray) -> np.ndarray:
    return normalize_algorithmic_confidence(probability)


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
        return float(
            np.sum(uncertainties * weights, dtype=np.float64) / max(EPS, float(np.sum(weights, dtype=np.float64)))
        )
    top_indices = np.argpartition(-uncertainties, count - 1)[:count]
    top_uncertainty = uncertainties[top_indices]
    top_weights = weights[top_indices]
    return float(
        np.sum(top_uncertainty * top_weights, dtype=np.float64) / max(EPS, float(np.sum(top_weights, dtype=np.float64)))
    )


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
    return (
        float(np.clip(score, 0.0, 1.0)),
        mean_uncertainty,
        low_conf_fraction,
        top_uncertainty_mean,
        largest_region_fraction,
    )


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


def _model_output_confidence_metrics(confidence_map: np.ndarray) -> ModelOutputConfidenceMetrics:
    confidence = np.clip(np.asarray(confidence_map, dtype=np.float32), 0.0, 1.0)
    if confidence.ndim != 2 or confidence.size == 0:
        return ModelOutputConfidenceMetrics(
            frame_uncertainty_score=0.0,
            mean_confidence=0.0,
            mean_uncertainty=0.0,
            uncertain_fraction=0.0,
            top_uncertainty_mean=0.0,
            largest_uncertain_region_fraction=0.0,
            min_confidence=0.0,
            max_confidence=0.0,
        )
    uncertainty = _uncertainty_map_from_probability(confidence)
    support_weights = np.ones_like(confidence, dtype=np.float32)
    score, mean_uncertainty, uncertain_fraction, top_uncertainty_mean, largest_region = (
        _frame_uncertainty_components_from_maps(
            uncertainty,
            support_weights,
            uncertainty_threshold=float(MODEL_RISK_UNCERTAINTY_THRESHOLD),
            top_fraction=float(MODEL_RISK_TOP_UNCERTAIN_FRACTION),
            risk_weight_mean=float(MODEL_RISK_WEIGHT_MEAN),
            risk_weight_fraction=float(MODEL_RISK_WEIGHT_FRACTION),
            risk_weight_top=float(MODEL_RISK_WEIGHT_TOP),
            risk_weight_cluster=float(MODEL_RISK_WEIGHT_CLUSTER),
        )
    )
    finite = confidence[np.isfinite(confidence)]
    if finite.size == 0:
        mean_confidence = min_confidence = max_confidence = 0.0
    else:
        mean_confidence = float(np.mean(finite, dtype=np.float64))
        min_confidence = float(np.min(finite))
        max_confidence = float(np.max(finite))
    return ModelOutputConfidenceMetrics(
        frame_uncertainty_score=float(score),
        mean_confidence=float(mean_confidence),
        mean_uncertainty=float(mean_uncertainty),
        uncertain_fraction=float(uncertain_fraction),
        top_uncertainty_mean=float(top_uncertainty_mean),
        largest_uncertain_region_fraction=float(largest_region),
        min_confidence=float(min_confidence),
        max_confidence=float(max_confidence),
    )


def _point_uncertainty_cluster_fraction(
    coordinates: tuple[tuple[float, float, float], ...],
    uncertainty_values: np.ndarray,
    support_mask: np.ndarray,
    *,
    uncertainty_threshold: float,
) -> float:
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    uncertainties = np.asarray(uncertainty_values, dtype=np.float64).reshape(-1)
    if not coordinates or support.size == 0 or not np.any(support):
        return 0.0
    uncertain_indices = [
        index
        for index, (is_supported, uncertainty) in enumerate(zip(support.tolist(), uncertainties.tolist()))
        if is_supported and np.isfinite(uncertainty) and float(uncertainty) > float(uncertainty_threshold)
    ]
    if not uncertain_indices:
        return 0.0
    support_count = int(np.count_nonzero(support))
    adjacency: dict[int, list[int]] = {index: [] for index in uncertain_indices}
    for offset, left_index in enumerate(uncertain_indices):
        left_x, left_y, left_radius = coordinates[left_index]
        left_radius = max(1.0, float(left_radius))
        for right_index in uncertain_indices[offset + 1 :]:
            right_x, right_y, right_radius = coordinates[right_index]
            right_radius = max(1.0, float(right_radius))
            max_distance = left_radius + right_radius + 1.0
            if math.hypot(float(left_x) - float(right_x), float(left_y) - float(right_y)) <= max_distance:
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)
    visited: set[int] = set()
    largest_cluster = 0
    for start_index in uncertain_indices:
        if start_index in visited:
            continue
        stack = [start_index]
        visited.add(start_index)
        cluster_size = 0
        while stack:
            current = stack.pop()
            cluster_size += 1
            for neighbor in adjacency.get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        largest_cluster = max(largest_cluster, cluster_size)
    return float(largest_cluster / max(1, support_count))


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
        support_weights[support_mask] = (probabilities[support_mask] - float(support_threshold)) / max(
            EPS, 1.0 - float(support_threshold)
        )
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
    return (
        float(np.clip(score, 0.0, 1.0)),
        mean_uncertainty,
        low_conf_fraction,
        top_uncertainty_mean,
        largest_region_fraction,
    )


def _probability_gradient(probability: np.ndarray) -> np.ndarray:
    """Compute simple central-difference gradient magnitude for a probability map."""

    prob = np.asarray(probability, dtype=np.float32)
    if prob.ndim != 2 or prob.size == 0:
        return np.zeros_like(prob, dtype=np.float32)
    padded = np.pad(prob, 1, mode="edge")
    grad_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    grad_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return np.hypot(grad_x, grad_y).astype(np.float32)


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
    boundary_uncertainty = (
        float(np.mean(uncertainty[boundary_mask], dtype=np.float64)) if np.any(boundary_mask) else 0.0
    )
    core_threshold = max(
        0.65, float(np.quantile(support_probability_values, 0.65)) if support_probability_values.size > 0 else 0.65
    )
    core_mask = support_mask & (prob >= core_threshold)
    core_confidence = float(np.mean(confidence[core_mask], dtype=np.float64)) if np.any(core_mask) else 0.0
    labels, polygon_count = _label_components(strong if np.any(strong) else support_mask)
    weight_total = float(np.sum(support_weight_values, dtype=np.float64))
    mean_confidence = (
        float(
            np.sum(support_weight_values * np.asarray(confidence[support_mask], dtype=np.float32), dtype=np.float64)
            / max(EPS, weight_total)
        )
        if weight_total > 0.0
        else 0.0
    )
    mean_probability = (
        float(np.sum(support_weight_values * support_probability_values, dtype=np.float64) / max(EPS, weight_total))
        if weight_total > 0.0
        else 0.0
    )
    frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = (
        _frame_uncertainty_components_from_maps(
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


def _polygon_confidence_config() -> PolygonConfidencePipelineConfig:
    """Return the default configuration for polygon confidence extraction."""

    return PolygonConfidencePipelineConfig(
        gaussian_sigma=float(POLYGON_CONFIDENCE_PREPROC_GAUSSIAN_SIGMA),
        median_radius=int(POLYGON_CONFIDENCE_PREPROC_MEDIAN_RADIUS),
        local_normalization_radius=int(POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_RADIUS),
        local_normalization_strength=float(POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_STRENGTH),
        hysteresis_low_ratio=float(POLYGON_CONFIDENCE_HYSTERESIS_LOW_RATIO),
        hysteresis_low_floor=float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR),
        elongated_vertical_radius=int(POLYGON_CONFIDENCE_ELONGATED_VERTICAL_RADIUS),
        elongated_horizontal_radius=int(POLYGON_CONFIDENCE_ELONGATED_HORIZONTAL_RADIUS),
        elongated_min_aspect_ratio=float(POLYGON_CONFIDENCE_ELONGATED_MIN_ASPECT_RATIO),
        elongated_min_area=int(POLYGON_CONFIDENCE_ELONGATED_MIN_AREA),
        dominant_min_area=int(POLYGON_CONFIDENCE_DOMINANT_MIN_AREA),
        dominant_min_mean_probability=float(POLYGON_CONFIDENCE_DOMINANT_MIN_MEAN_PROBABILITY),
        dominant_min_aspect_ratio=float(POLYGON_CONFIDENCE_DOMINANT_MIN_ASPECT_RATIO),
        dominant_min_extent=float(POLYGON_CONFIDENCE_DOMINANT_MIN_EXTENT),
        dominant_large_area=int(POLYGON_CONFIDENCE_DOMINANT_LARGE_AREA),
        dominant_lock_radius=int(POLYGON_CONFIDENCE_DOMINANT_LOCK_RADIUS),
        large_polygon_low_scale=float(POLYGON_CONFIDENCE_LARGE_POLYGON_LOW_SCALE),
        large_polygon_min_area=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_AREA),
        large_polygon_min_major_span=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_MAJOR_SPAN),
        large_polygon_min_extent=float(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_EXTENT),
        large_polygon_min_aspect_ratio=float(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_ASPECT_RATIO),
        large_polygon_band_expand=int(POLYGON_CONFIDENCE_LARGE_POLYGON_BAND_EXPAND),
        large_polygon_roi_padding=int(POLYGON_CONFIDENCE_LARGE_POLYGON_ROI_PADDING),
        large_polygon_seed_low_scale=float(POLYGON_CONFIDENCE_LARGE_POLYGON_SEED_LOW_SCALE),
        large_polygon_major_close_radius=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MAJOR_CLOSE_RADIUS),
        large_polygon_minor_close_radius=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MINOR_CLOSE_RADIUS),
        large_polygon_barrier_delta=float(POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_DELTA),
        large_polygon_barrier_coverage_min=float(POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_COVERAGE_MIN),
        small_low_scale=float(POLYGON_CONFIDENCE_SMALL_LOW_SCALE),
        small_high_scale=float(POLYGON_CONFIDENCE_SMALL_HIGH_SCALE),
        small_min_area=max(1, int(POLYGON_CONFIDENCE_PROPOSAL_MIN_AREA)),
        small_max_area=int(POLYGON_CONFIDENCE_SMALL_MAX_AREA),
        small_peak_floor=float(POLYGON_CONFIDENCE_PROPOSAL_PEAK_FLOOR),
        small_mean_floor=float(POLYGON_CONFIDENCE_PROPOSAL_MEAN_FLOOR),
        adaptive_radius=int(POLYGON_CONFIDENCE_ADAPTIVE_RADIUS),
        adaptive_low_offset=float(POLYGON_CONFIDENCE_ADAPTIVE_LOW_OFFSET),
        adaptive_high_offset=float(POLYGON_CONFIDENCE_ADAPTIVE_HIGH_OFFSET),
        separation_core_min_area=int(POLYGON_CONFIDENCE_SEPARATION_CORE_MIN_AREA),
        separation_roi_padding=int(POLYGON_CONFIDENCE_SEPARATION_ROI_PADDING),
        separation_boundary_low_weight=float(POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_LOW_WEIGHT),
        separation_boundary_contrast_weight=float(POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_CONTRAST_WEIGHT),
        separation_boundary_uncertainty_weight=float(POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_UNCERTAINTY_WEIGHT),
        separation_barrier_threshold=float(POLYGON_CONFIDENCE_SEPARATION_BARRIER_THRESHOLD),
        separation_barrier_dilate_radius=int(POLYGON_CONFIDENCE_SEPARATION_BARRIER_DILATE_RADIUS),
        separation_bridge_probability_max=float(POLYGON_CONFIDENCE_SEPARATION_BRIDGE_PROBABILITY_MAX),
        separation_bridge_barrier_threshold=float(POLYGON_CONFIDENCE_SEPARATION_BRIDGE_BARRIER_THRESHOLD),
        merge_iou_threshold=float(POLYGON_CONFIDENCE_MERGE_IOU_THRESHOLD),
        merge_distance=int(POLYGON_CONFIDENCE_MERGE_DISTANCE),
        enable_watershed=bool(POLYGON_CONFIDENCE_ENABLE_WATERSHED),
        watershed_seed_min_area=int(POLYGON_CONFIDENCE_WATERSHED_SEED_MIN_AREA),
        hole_probability_scale=float(POLYGON_CONFIDENCE_HOLE_PROBABILITY_SCALE),
        hole_probability_max=float(POLYGON_CONFIDENCE_HOLE_PROBABILITY_MAX),
        hole_min_area=int(POLYGON_CONFIDENCE_HOLE_MIN_AREA),
        spill_large_area_fraction=float(POLYGON_CONFIDENCE_SPILL_LARGE_AREA_FRACTION),
        spill_large_extent=float(POLYGON_CONFIDENCE_SPILL_LARGE_EXTENT),
        spill_low_texture_max=float(POLYGON_CONFIDENCE_SPILL_LOW_TEXTURE_MAX),
        spill_trim_delta=float(POLYGON_CONFIDENCE_SPILL_TRIM_DELTA),
        spill_boundary_separation_max=float(POLYGON_CONFIDENCE_SPILL_BOUNDARY_SEPARATION_MAX),
        spill_peak_margin_max=float(POLYGON_CONFIDENCE_SPILL_PEAK_MARGIN_MAX),
        spill_ribbon_aspect_min=float(POLYGON_CONFIDENCE_SPILL_RIBBON_ASPECT_MIN),
        spill_border_coverage_min=float(POLYGON_CONFIDENCE_SPILL_BORDER_COVERAGE_MIN),
        spill_mean_probability_max=float(POLYGON_CONFIDENCE_SPILL_MEAN_PROBABILITY_MAX),
        spill_cross_axis_max=float(POLYGON_CONFIDENCE_SPILL_CROSS_AXIS_MAX),
        spill_prominence_min=float(POLYGON_CONFIDENCE_SPILL_PROMINENCE_MIN),
        spill_strong_axis_coverage_min=float(POLYGON_CONFIDENCE_SPILL_STRONG_AXIS_COVERAGE_MIN),
        spill_strong_area_fraction_min=float(POLYGON_CONFIDENCE_SPILL_STRONG_AREA_FRACTION_MIN),
        boundary_snap_min_aspect=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_ASPECT),
        boundary_snap_profile_quantile=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_PROFILE_QUANTILE),
        boundary_snap_min_drop=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_DROP),
        boundary_snap_min_retained_fraction=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_RETAINED_FRACTION),
        valley_minor_coverage_min=float(POLYGON_CONFIDENCE_VALLEY_MINOR_COVERAGE_MIN),
    )


def _point_confidence_patch(array: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    patch_radius = max(0, int(radius))
    px = int(round(float(x)))
    py = int(round(float(y)))
    if array.ndim != 2 or array.size == 0 or py < 0 or py >= array.shape[0] or px < 0 or px >= array.shape[1]:
        return np.zeros((0, 0), dtype=np.float32)
    y0 = max(0, py - patch_radius)
    y1 = min(array.shape[0], py + patch_radius + 1)
    x0 = max(0, px - patch_radius)
    x1 = min(array.shape[1], px + patch_radius + 1)
    return np.asarray(array[y0:y1, x0:x1], dtype=np.float32)


def _point_local_contrast(
    probability: np.ndarray, x: float, y: float, radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS
) -> float:
    prob = np.asarray(probability, dtype=np.float32)
    px = int(round(float(x)))
    py = int(round(float(y)))
    if prob.ndim != 2 or prob.size == 0 or py < 0 or py >= prob.shape[0] or px < 0 or px >= prob.shape[1]:
        return 0.0
    patch = _point_confidence_patch(prob, x, y, radius)
    if patch.size <= 1:
        return 0.0
    center_value = float(prob[py, px])
    patch_sum = float(np.sum(patch, dtype=np.float64)) - center_value
    neighbor_count = max(1, int(patch.size - 1))
    return float(center_value - patch_sum / float(neighbor_count))


def _point_internal_confidence(
    prediction_view: object,
    *,
    neighborhood_radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    include_objects: bool = True,
) -> PointConfidenceMetrics:
    probability = _internal_confidence_probability_map(
        _prob_from_gray(np.asarray(getattr(prediction_view, "pred_gray"), dtype=np.uint8)),
        support_mask=np.asarray(getattr(prediction_view, "pred_bin"), dtype=bool),
    )
    confidence_map = _confidence_map_from_probability(probability)
    points = tuple(getattr(prediction_view, "points", ()))
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
        x = float(getattr(point, "x", 0.0))
        y = float(getattr(point, "y", 0.0))
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
        radius = max(1.0, float(getattr(point, "radius", 0.0)))
        if include_objects:
            object_rows.append(
                PointObjectConfidence(
                    object_id=int(index),
                    x=x,
                    y=y,
                    radius=radius,
                    point_probability=point_prob,
                    center_confidence=center_confidence,
                    local_confidence=local_confidence,
                    local_contrast=local_contrast,
                )
            )
        center_confidences.append(center_confidence)
        local_confidences.append(local_confidence)
        point_probs.append(point_prob)
        point_contrasts.append(local_contrast)
        point_weights.append(
            float(
                max(
                    0.0,
                    min(
                        1.0,
                        (point_prob - float(POINT_SUPPORT_THRESHOLD)) / max(EPS, 1.0 - float(POINT_SUPPORT_THRESHOLD)),
                    ),
                )
            )
            if point_prob >= float(POINT_SUPPORT_THRESHOLD)
            else 0.0
        )
        point_coordinates.append((x, y, radius))
    confidence_array = np.asarray(center_confidences, dtype=np.float64)
    weight_array = np.asarray(point_weights, dtype=np.float64)
    weight_sum = float(np.sum(weight_array, dtype=np.float64))
    frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = (
        _frame_uncertainty_components_from_points(
            np.asarray(point_probs, dtype=np.float32),
            tuple(point_coordinates),
            support_threshold=POINT_SUPPORT_THRESHOLD,
        )
    )
    return PointConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        mean_uncertainty=mean_uncertainty,
        uncertain_support_fraction=low_conf_fraction,
        top_uncertainty_mean=worst_tail_uncertainty,
        largest_uncertain_region_fraction=largest_low_conf_component,
        mean_point_confidence=float(np.sum(confidence_array * weight_array, dtype=np.float64) / max(EPS, weight_sum))
        if weight_sum > 0.0
        else 0.0,
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
