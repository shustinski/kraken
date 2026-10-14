"""repository: analyze record payload."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    FrameRecord,
    GeometryMode,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _clip01

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.build_model_payloads import _build_model_payloads

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.internal_confidence_probability_map import _store_cached_record_payload

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.pairwise_model_comparisons import _pairwise_model_comparisons

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.record_payload_cache_key import (
    _clear_runtime_image_caches,
    _load_cached_record_payload,
    _record_payload_cache_key,
)

def _analyze_record_payload(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    cache_enabled: bool,
    include_model_confidence: bool = True,
    include_pairwise_metrics: bool = True,
    include_model_metrics: bool = True,
) -> dict[str, object] | None:
    timings_ms: dict[str, float] = {}
    cache_key = None
    if cache_enabled:
        cache_key = _record_payload_cache_key(
            record,
            model_specs,
            analysis_max_side,
            geometry_mode,
            point_match_radius,
            boundary_radius,
            confidence_uncertainty_delta,
            point_confidence_radius,
            polygon_confidence_summary,
            include_model_confidence=bool(include_model_confidence),
            include_pairwise_metrics=bool(include_pairwise_metrics),
            include_model_metrics=bool(include_model_metrics),
        )
        cached = _load_cached_record_payload(cache_key)
        if cached is not None:
            return cached

    load_started = perf_counter()
    probabilities_by_model, masks_by_model, model_structures, model_diagnostics, model_metrics, model_confidence, model_confidence_output_available, _original_gray, gt_mask, _gt_structure, resolved_geometry_mode, model_views, gt_point_view = _build_model_payloads(
        record,
        model_specs,
        analysis_max_side=analysis_max_side,
        geometry_mode=geometry_mode,
        point_match_radius=point_match_radius,
        boundary_radius=boundary_radius,
        confidence_uncertainty_delta=confidence_uncertainty_delta,
        point_confidence_radius=point_confidence_radius,
        polygon_confidence_summary=polygon_confidence_summary,
        include_confidence_objects=False,
        include_original_gray=False,
        include_model_confidence=include_model_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
        include_model_metrics=include_model_metrics,
        include_model_diagnostics=bool(include_pairwise_metrics or include_model_metrics),
        include_structure_details=False,
    )
    timings_ms['loading_preprocess'] = 1000.0 * (perf_counter() - load_started)
    probabilities = list(probabilities_by_model.values())
    if not probabilities:
        return None

    metrics_started = perf_counter()
    pairwise_rows: tuple[dict[str, object], ...] = ()
    disagreement = 0.0
    model_model_score = 1.0
    if include_pairwise_metrics:
        pairwise_rows = _pairwise_model_comparisons(
            probabilities_by_model,
            masks_by_model,
            geometry_mode=resolved_geometry_mode,
            model_views=model_views,
            model_structures=model_structures,
            point_match_radius=point_match_radius,
        )
        agreement_scores = np.asarray([float(row.get('agreement_score', 0.0)) for row in pairwise_rows], dtype=np.float64)
        disagreement = float(np.mean(1.0 - agreement_scores, dtype=np.float64)) if agreement_scores.size else 0.0
        model_model_score = float(np.mean(agreement_scores, dtype=np.float64)) if agreement_scores.size else 1.0

    vector: dict[str, float] = {}

    if model_metrics:
        qualities = np.asarray([float(getattr(metrics, 'quality_score', 0.0)) for metrics in model_metrics.values()], dtype=np.float64)
        labeled_best = float(np.max(qualities))
        labeled_mean = float(np.mean(qualities, dtype=np.float64))
        model_labeled_score = float(labeled_best)
    else:
        labeled_best = None
        labeled_mean = None
        model_labeled_score = None
    timings_ms['metrics'] = 1000.0 * (perf_counter() - metrics_started)

    assemble_started = perf_counter()
    payload = {
        'key': record.key,
        'geometry_mode': resolved_geometry_mode.value,
        'vector': vector,
        'disagreement': float(disagreement),
        'model_model_score': float(_clip01(model_model_score)),
        'labeled_best': labeled_best,
        'labeled_mean': labeled_mean,
        'model_labeled_score': None if model_labeled_score is None else float(_clip01(model_labeled_score)),
        'model_diagnostics': model_diagnostics,
        'model_metrics': model_metrics,
        'model_confidence': model_confidence,
        'model_confidence_output_available': model_confidence_output_available,
        'pairwise_rows': pairwise_rows,
        'timings_ms': timings_ms,
    }
    timings_ms['payload_assembly'] = 1000.0 * (perf_counter() - assemble_started)
    if cache_enabled and cache_key is not None:
        _store_cached_record_payload(cache_key, payload)
    return payload

def _analyze_record_payload_for_executor(args: tuple[FrameRecord, tuple[ModelSpec, ...], int | None, GeometryMode, float, int, float, int, str, bool, bool, bool, bool]) -> tuple[str, dict[str, object] | None]:
    record, model_specs, analysis_max_side, geometry_mode, point_match_radius, boundary_radius, confidence_uncertainty_delta, point_confidence_radius, polygon_confidence_summary, cache_enabled, include_model_confidence, include_pairwise_metrics, include_model_metrics = args
    return record.key, _analyze_record_payload(
        record,
        model_specs,
        analysis_max_side,
        geometry_mode,
        point_match_radius,
        boundary_radius,
        confidence_uncertainty_delta,
        point_confidence_radius,
        polygon_confidence_summary,
        cache_enabled,
        include_model_confidence,
        include_pairwise_metrics,
        include_model_metrics,
    )

def _analyze_record_payload_batch_for_executor(
    batch_args: tuple[tuple[FrameRecord, tuple[ModelSpec, ...], int | None, GeometryMode, float, int, float, int, str, bool, bool, bool, bool], ...]
) -> tuple[tuple[str, dict[str, object] | None], ...]:
    results: list[tuple[str, dict[str, object] | None]] = []
    for args in batch_args:
        results.append(_analyze_record_payload_for_executor(args))
    _clear_runtime_image_caches()
    return tuple(results)
