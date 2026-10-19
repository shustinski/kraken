"""repository: polygon internal confidence."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    POLYGON_SUPPORT_THRESHOLD,
    POLYGON_CONFIDENCE_SUMMARY_CORE,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidenceMetrics,
    PolygonConfidencePipelineConfig,
    PolygonObjectConfidence,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _weighted_mean

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.frame_uncertainty_components_maps import _frame_uncertainty_components_from_probability

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.internal_confidence_probability_map import (
    _confidence_map_from_probability,
    _internal_confidence_probability_map,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_erode,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_geometry import (
    _mask_bbox,
    _mask_geometry,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _polygon_confidence_config

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_pipeline import _polygon_confidence_pipeline

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_transition_uncertainty_maps import _polygon_transition_uncertainty_maps

def _polygon_internal_confidence(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    boundary_radius: int = 1,
    uncertainty_delta: float = MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    summary_metric: str = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    strong_threshold: float = 0.5,
    include_objects: bool = True,
    include_debug: bool = False,
    config: PolygonConfidencePipelineConfig | None = None,
) -> PolygonConfidenceMetrics:
    from neuralimage.Validation_gradient_widget_lite.core.repository import ndi

    prob = _internal_confidence_probability_map(probability, support_mask=np.asarray(mask, dtype=bool))
    pipeline_config = config or _polygon_confidence_config()
    mask_bool, final_candidates, debug_data = _polygon_confidence_pipeline(
        prob,
        np.asarray(mask, dtype=bool),
        strong_threshold=float(strong_threshold),
        config=pipeline_config,
        include_debug=include_debug,
    )
    confidence_map = _confidence_map_from_probability(prob)
    object_count = int(np.count_nonzero(mask_bool))
    area_fraction = float(object_count / max(1, mask_bool.size))
    frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = _frame_uncertainty_components_from_probability(
        prob,
        support_threshold=POLYGON_SUPPORT_THRESHOLD,
    )
    normalized_summary_metric = str(summary_metric or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED).strip().lower()
    if normalized_summary_metric != POLYGON_CONFIDENCE_SUMMARY_CORE:
        normalized_summary_metric = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
    if object_count <= 0:
        return PolygonConfidenceMetrics(
            frame_uncertainty_score=0.0,
            mean_uncertainty=0.0,
            uncertain_support_fraction=0.0,
            top_uncertainty_mean=0.0,
            largest_uncertain_region_fraction=0.0,
            mean_object_confidence=0.0,
            mean_core_confidence=0.0,
            mean_boundary_uncertainty=0.0,
            mean_weighted_confidence=0.0,
            mean_object_probability=0.0,
            uncertain_fraction=0.0,
            mean_transition_width=0.0,
            object_area_fraction=area_fraction,
            polygon_count=0,
            summary_metric=normalized_summary_metric,
            low_conf_fraction=0.0,
            worst_tail_uncertainty=0.0,
            largest_low_conf_component=0.0,
            objects=(),
            debug_data=debug_data if include_debug else None,
        )

    object_rows: list[PolygonObjectConfidence] = []
    aggregate_summary: list[tuple[float, float]] = []
    aggregate_core: list[tuple[float, float]] = []
    aggregate_boundary: list[tuple[float, float]] = []
    aggregate_weighted: list[tuple[float, float]] = []
    aggregate_probability: list[tuple[float, float]] = []
    aggregate_uncertain: list[tuple[float, float]] = []
    aggregate_transition_width: list[tuple[float, float]] = []
    object_total = 0
    delta = float(max(EPS, uncertainty_delta))
    erosion_radius = max(0, int(boundary_radius))
    transition_width_map, _inverse_local_contrast, transition_uncertainty_map = _polygon_transition_uncertainty_maps(
        prob,
        mask_bool,
        contrast_radius=max(1, erosion_radius or 1),
    )

    candidate_entries: list[tuple[int, np.ndarray, tuple[str, ...], tuple[int, int, int, int], float, float]] = []
    if final_candidates:
        for object_id, candidate in enumerate(final_candidates, start=1):
            object_mask = np.asarray(candidate.mask, dtype=bool) & mask_bool
            if not np.any(object_mask):
                continue
            candidate_entries.append((
                int(object_id),
                object_mask,
                tuple(candidate.source_branches) or ('merged',),
                tuple(candidate.bbox),
                float(candidate.aspect_ratio),
                float(candidate.elongation),
            ))
    else:
        labels, label_count = _label_components(mask_bool)
        for label_id in range(1, int(label_count) + 1):
            object_mask = labels == label_id
            if not np.any(object_mask):
                continue
            bbox = _mask_bbox(object_mask)
            _area, _bbox, aspect_ratio, elongation, _extent = _mask_geometry(object_mask)
            candidate_entries.append((
                int(label_id),
                object_mask,
                ('merged',),
                tuple(bbox),
                float(aspect_ratio),
                float(elongation),
            ))

    for object_id, object_mask, source_branches, bbox, aspect_ratio, elongation in candidate_entries:
        area = int(np.count_nonzero(object_mask))
        if area <= 0:
            continue
        object_total += 1
        morphological_interior = _binary_erode(object_mask, erosion_radius) & object_mask if erosion_radius > 0 else object_mask.copy()

        object_conf = np.asarray(confidence_map[object_mask], dtype=np.float32)
        object_prob = np.asarray(prob[object_mask], dtype=np.float32)
        object_peak = float(np.max(object_prob)) if object_prob.size > 0 else 0.0
        object_mean = float(np.mean(object_prob, dtype=np.float64)) if object_prob.size > 0 else 0.0
        plateau_threshold = max(0.5, object_mean + 0.25 * max(0.0, object_peak - object_mean))
        plateau_mask = object_mask & (prob >= plateau_threshold)
        if np.any(morphological_interior & plateau_mask):
            interior = morphological_interior & plateau_mask
        elif np.any(plateau_mask):
            interior = plateau_mask
        elif np.any(morphological_interior):
            interior = morphological_interior
        else:
            interior = object_mask.copy()

        boundary_band = object_mask & np.logical_not(interior)
        if not np.any(boundary_band):
            boundary_band = object_mask.copy()

        core_confidence = float(np.mean(confidence_map[interior], dtype=np.float64))
        mean_confidence = float(np.mean(object_conf, dtype=np.float64)) if object_conf.size > 0 else 0.0
        median_confidence = float(np.median(object_conf)) if object_conf.size > 0 else 0.0
        min_confidence = float(np.min(object_conf)) if object_conf.size > 0 else 0.0
        max_confidence = float(np.max(object_conf)) if object_conf.size > 0 else 0.0
        low_percentile_confidence = float(np.percentile(object_conf, 25.0)) if object_conf.size > 0 else 0.0
        boundary_transition_uncertainty = np.asarray(transition_uncertainty_map[boundary_band], dtype=np.float32)
        boundary_uncertainty = float(np.mean(boundary_transition_uncertainty, dtype=np.float64))
        boundary_transition_width = np.asarray(transition_width_map[boundary_band], dtype=np.float32)
        if boundary_transition_width.size > 0 and boundary_transition_uncertainty.size > 0:
            transition_width_mean = float(np.mean(boundary_transition_width * boundary_transition_uncertainty, dtype=np.float64))
        else:
            transition_width_mean = 0.0
        if ndi is not None:
            weights = np.asarray(ndi.distance_transform_edt(object_mask), dtype=np.float32)[object_mask]
        else:
            weights = np.ones(area, dtype=np.float32)
        if weights.size == 0 or float(np.max(weights)) <= EPS:
            normalized_weights = np.ones(area, dtype=np.float32)
        else:
            normalized_weights = np.clip(weights / float(np.max(weights)), EPS, 1.0).astype(np.float32)
        weighted_confidence = float(np.average(object_conf, weights=normalized_weights))
        mean_probability = float(np.mean(object_prob, dtype=np.float64))
        uncertain_fraction = float(np.mean(np.abs(object_prob - 0.5) < delta, dtype=np.float64))
        ys, xs = np.nonzero(object_mask)
        centroid_x = float(np.mean(xs, dtype=np.float64)) if xs.size else 0.0
        centroid_y = float(np.mean(ys, dtype=np.float64)) if ys.size else 0.0
        source_branch = source_branches[0] if len(source_branches) == 1 else 'merged'
        summary_confidence = weighted_confidence if normalized_summary_metric == POLYGON_CONFIDENCE_SUMMARY_WEIGHTED else core_confidence
        if include_objects:
            object_rows.append(PolygonObjectConfidence(
                object_id=int(object_id),
                area=area,
                area_fraction=float(area / max(1, mask_bool.size)),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                core_confidence=core_confidence,
                boundary_uncertainty=boundary_uncertainty,
                weighted_confidence=weighted_confidence,
                summary_confidence=summary_confidence,
                mean_probability=mean_probability,
                mean_confidence=mean_confidence,
                median_confidence=median_confidence,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                low_percentile_confidence=low_percentile_confidence,
                uncertain_fraction=uncertain_fraction,
                transition_width_mean=transition_width_mean,
                bbox_x=int(bbox[0]),
                bbox_y=int(bbox[1]),
                bbox_width=int(bbox[2]),
                bbox_height=int(bbox[3]),
                aspect_ratio=float(aspect_ratio),
                elongation=float(elongation),
                source_branch=source_branch,
                source_branches=tuple(source_branches),
            ))
        weight = float(area)
        aggregate_summary.append((summary_confidence, weight))
        aggregate_core.append((core_confidence, weight))
        aggregate_boundary.append((boundary_uncertainty, weight))
        aggregate_weighted.append((weighted_confidence, weight))
        aggregate_probability.append((mean_probability, weight))
        aggregate_uncertain.append((uncertain_fraction, weight))
        aggregate_transition_width.append((transition_width_mean, weight))

    return PolygonConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        mean_uncertainty=mean_uncertainty,
        uncertain_support_fraction=low_conf_fraction,
        top_uncertainty_mean=worst_tail_uncertainty,
        largest_uncertain_region_fraction=largest_low_conf_component,
        mean_object_confidence=_weighted_mean(aggregate_summary),
        mean_core_confidence=_weighted_mean(aggregate_core),
        mean_boundary_uncertainty=_weighted_mean(aggregate_boundary),
        mean_weighted_confidence=_weighted_mean(aggregate_weighted),
        mean_object_probability=_weighted_mean(aggregate_probability),
        uncertain_fraction=_weighted_mean(aggregate_uncertain),
        mean_transition_width=_weighted_mean(aggregate_transition_width),
        object_area_fraction=area_fraction,
        polygon_count=int(object_total),
        summary_metric=normalized_summary_metric,
        low_conf_fraction=low_conf_fraction,
        worst_tail_uncertainty=worst_tail_uncertainty,
        largest_low_conf_component=largest_low_conf_component,
        objects=tuple(object_rows),
        debug_data=debug_data if include_debug else None,
    )
