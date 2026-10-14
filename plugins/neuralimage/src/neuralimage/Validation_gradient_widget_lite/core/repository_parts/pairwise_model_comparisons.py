"""repository: pairwise model comparisons."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    GeometryMode,
    PointDiagnosticMetrics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import (
    _clip01,
    _normalize_ratio,
    _weighted_mean,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import (
    _dice,
    _mask_structure,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.pairwise_mask_metrics import _pairwise_mask_metrics

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.point_model_agreement import (
    _point_model_agreement,
    _prepare_mask_pairwise_descriptors,
)

def _pairwise_model_comparisons(
    probabilities_by_model: dict[str, np.ndarray],
    masks_by_model: dict[str, np.ndarray],
    *,
    geometry_mode: GeometryMode = GeometryMode.MASK,
    model_views: dict[str, object] | None = None,
    model_structures: dict[str, dict[str, object]] | None = None,
    point_match_radius: float = 3.0,
) -> tuple[dict[str, object], ...]:
    model_ids = list(probabilities_by_model.keys())
    rows: list[dict[str, object]] = []
    current_views = model_views or {}
    mask_descriptors = _prepare_mask_pairwise_descriptors(probabilities_by_model, masks_by_model, model_structures=model_structures) if geometry_mode != GeometryMode.POINT else {}
    for index_a, model_a in enumerate(model_ids):
        for model_b in model_ids[index_a + 1:]:
            if geometry_mode == GeometryMode.POINT and model_a in current_views and model_b in current_views:
                metrics = _point_model_agreement(current_views[model_a], current_views[model_b], point_match_radius)
                rows.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'precision': float(metrics.precision_at_radius),
                    'recall': float(metrics.recall_at_radius),
                    'f1': float(metrics.f1_at_radius),
                    'mean_localization_error': float(metrics.mean_localization_error),
                    'localization_agreement': float(metrics.localization_agreement),
                    'count_agreement': float(metrics.count_agreement),
                    'matched_count': int(metrics.matched_count),
                    'tp': int(metrics.true_positive_count),
                    'fp': int(metrics.false_positive_count),
                    'fn': int(metrics.false_negative_count),
                    'point_count_a': int(metrics.point_count_a),
                    'point_count_b': int(metrics.point_count_b),
                    'agreement_score': float(metrics.agreement_score),
                })
                continue

            metrics_row = _pairwise_mask_metrics(mask_descriptors[str(model_a)], mask_descriptors[str(model_b)])
            rows.append({
                'model_a': model_a,
                'model_b': model_b,
                **metrics_row,
            })
    return tuple(rows)

def _point_feature_vector(prediction_view: object) -> dict[str, float]:
    points = tuple(getattr(prediction_view, "points", ()))
    point_count = len(points)
    image_shape = tuple(int(v) for v in getattr(prediction_view, "pred_gray").shape)
    image_area = max(1.0, float(image_shape[0] * image_shape[1]))
    mean_radius = float(np.mean([float(getattr(point, "radius", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    mean_peak_intensity = float(np.mean([float(getattr(point, "peak_intensity", 0.0)) for point in points], dtype=np.float64) / 255.0) if points else 0.0
    mean_local_snr = float(np.mean([float(getattr(point, "local_snr", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    mean_blob_score = float(np.mean([float(getattr(point, "blob_score", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    return {
        "area_fraction": float(point_count / image_area),
        "component_count": float(point_count),
        "mean_component_area": float(mean_peak_intensity),
        "skeleton_length": float(mean_radius),
        "endpoint_count": float(_normalize_ratio(mean_local_snr)),
        "branchpoint_count": float(_normalize_ratio(mean_blob_score)),
    }

def _point_diagnostic_metrics(prediction_view: object) -> PointDiagnosticMetrics:
    points = tuple(getattr(prediction_view, "points", ()))
    mean_radius = float(np.mean([float(getattr(point, "radius", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    mean_peak_intensity = float(np.mean([float(getattr(point, "peak_intensity", 0.0)) for point in points], dtype=np.float64) / 255.0) if points else 0.0
    mean_local_snr = float(np.mean([float(getattr(point, "local_snr", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    false_spot_ratio = float(_clip01(np.mean([1.0 if float(getattr(point, "local_snr", 0.0)) < 0.5 else 0.0 for point in points], dtype=np.float64))) if points else 0.0
    proxy_score = _weighted_mean([
        (_clip01(mean_peak_intensity), 0.45),
        (_clip01(_normalize_ratio(mean_local_snr)), 0.35),
        (1.0 - false_spot_ratio, 0.20),
    ])
    return PointDiagnosticMetrics(
        point_count=int(len(points)),
        mean_radius=float(mean_radius),
        mean_peak_intensity=float(mean_peak_intensity),
        false_spot_ratio=float(false_spot_ratio),
        proxy_score=float(proxy_score),
    )

def _infer_geometry_mode(prediction_view: object) -> GeometryMode:
    points = tuple(getattr(prediction_view, "points", ()))
    region_summary = getattr(prediction_view, "region_summary", None)
    area_fraction = float(getattr(region_summary, "area_fraction", 0.0)) if region_summary is not None else 0.0
    mean_area = float(getattr(region_summary, "mean_area", 0.0)) if region_summary is not None else 0.0
    if len(points) > 0 and area_fraction <= 0.08 and mean_area <= 64.0:
        return GeometryMode.POINT
    return GeometryMode.MASK

def _consensus_probability(probabilities: list[np.ndarray]) -> np.ndarray:
    if not probabilities:
        return np.zeros((1, 1), dtype=np.float32)
    stacked = np.stack(probabilities, axis=0).astype(np.float32)
    return np.mean(stacked, axis=0, dtype=np.float32)

def _entropy_map(probability: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    entropy = -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob)) / np.log(2.0)
    return np.asarray(entropy, dtype=np.float32)

def _disagreement_score(probabilities: list[np.ndarray], masks: list[np.ndarray]) -> tuple[float, np.ndarray]:
    if not probabilities:
        return 0.0, np.zeros((1, 1), dtype=np.float32)
    if len(probabilities) == 1:
        variance_map = np.zeros_like(probabilities[0], dtype=np.float32)
        return 0.0, variance_map
    pairwise: list[float] = []
    for index_a in range(len(masks)):
        for index_b in range(index_a + 1, len(masks)):
            pairwise.append(1.0 - _dice(masks[index_a], masks[index_b]))
    stacked = np.stack(probabilities, axis=0).astype(np.float32)
    variance_map = np.clip(4.0 * np.var(stacked, axis=0, dtype=np.float32), 0.0, 1.0)
    score = 0.6 * float(np.mean(pairwise, dtype=np.float64)) + 0.4 * float(np.mean(variance_map, dtype=np.float64))
    return float(_clip01(score)), variance_map.astype(np.float32)

def _structural_feature_vector(consensus_mask: np.ndarray, *, structure: dict[str, object] | None = None) -> dict[str, float]:
    current_structure = structure or _mask_structure(consensus_mask)
    return {
        "area_fraction": float(current_structure["area_fraction"]),
        "component_count": float(current_structure["component_count"]),
        "mean_component_area": float(current_structure["mean_component_area"]),
        "skeleton_length": float(current_structure["skeleton_length"]),
        "endpoint_count": float(current_structure["endpoint_count"]),
        "branchpoint_count": float(current_structure["branchpoint_count"]),
    }

def _robust_feature_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad if mad > EPS else float(np.std(array) + EPS)

def _structural_anomaly_score(vector: dict[str, float], bounds: dict[str, tuple[float, float]]) -> float:
    terms: list[float] = []
    for key, value in vector.items():
        median, scale = bounds.get(key, (0.0, 1.0))
        z_score = abs(float(value) - median) / max(EPS, scale)
        terms.append(_clip01(z_score / 3.0))
    return float(np.mean(terms, dtype=np.float64)) if terms else 0.0
