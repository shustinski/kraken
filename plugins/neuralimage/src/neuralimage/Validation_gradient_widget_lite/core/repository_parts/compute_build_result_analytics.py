"""repository: compute build result analytics."""
from __future__ import annotations
from dataclasses import replace
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildResult,
    FrameAnalysisSummary,
    FrameRecord,
    GeometryMode,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.aggregate_inter_model_point import (
    _aggregate_inter_model_point_scores,
    _aggregate_inter_model_polygon_scores,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _clip01

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.compute_record_payloads import (
    _aggregate_model_ranking,
    _available_metric_keys_for_models,
    _model_metric_key,
    _record_metric_value,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.iter_record_payloads import _iter_record_payloads

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.metric_higher_better import (
    compute_metric_percentiles,
    metric_higher_is_better,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.metric_requires_pairwise_metrics import (
    _metric_requires_model_confidence,
    _metric_requires_model_metrics,
    _metric_requires_pairwise_metrics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.record_payload_cache_key import _clear_runtime_image_caches

def compute_build_result_analytics(build_result: BuildResult, *, metric_key: str | None = None, progress_callback=None, state_callback=None, cancel_check=None) -> BuildResult:
    _clear_runtime_image_caches()
    records = list(build_result.records)
    if not records:
        return replace(build_result, scores_computed=True)

    active_metric = metric_key or build_result.selected_metric_key or 'overall_frame_score'
    if _metric_requires_model_confidence(active_metric) and active_metric not in set(build_result.available_metric_keys or ()):
        active_metric = 'overall_frame_score'
    include_model_confidence = _metric_requires_model_confidence(active_metric)
    include_pairwise_metrics = _metric_requires_pairwise_metrics(active_metric)
    include_model_metrics = _metric_requires_model_metrics(active_metric)

    records_by_key = {str(record.key): record for record in records}
    updated_records_by_key: dict[str, FrameRecord] = {}
    try:
        payload_iter = _iter_record_payloads(
            records,
            build_result.model_specs,
            build_result.options.analysis_max_side,
            build_result.options.max_workers,
            geometry_mode=build_result.options.geometry_mode,
            point_match_radius=float(build_result.options.point_match_radius),
            boundary_radius=int(getattr(build_result.options, 'boundary_radius', 1) or 1),
            confidence_uncertainty_delta=float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
            point_confidence_radius=int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS),
            polygon_confidence_summary=str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED),
            cache_enabled=bool(build_result.options.cache_enabled),
            include_model_confidence=include_model_confidence,
            include_pairwise_metrics=include_pairwise_metrics,
            include_model_metrics=include_model_metrics,
            progress_callback=progress_callback,
            state_callback=state_callback,
            cancel_check=cancel_check,
        )
        for record_key, payload in payload_iter:
            if payload is None:
                continue
            record = records_by_key.get(str(record_key))
            if record is None:
                continue
            vector = dict(payload.get('vector') or {})
            disagreement = float(payload.get('disagreement', 0.0))
            model_model_score = float(payload.get('model_model_score', 0.0))
            model_labeled_score = payload.get('model_labeled_score')
            frame_type = 'point' if str(payload.get('geometry_mode')) == GeometryMode.POINT.value else 'polygon'
            labeled_best = payload.get('labeled_best')
            labeled_mean = payload.get('labeled_mean')
            model_diagnostics = payload.get('model_diagnostics') or {}
            model_metrics = payload.get('model_metrics') or {}
            model_confidence = payload.get('model_confidence') or {}
            pairwise_rows = tuple(payload.get('pairwise_rows', ()))
            polygon_scores = _aggregate_inter_model_polygon_scores(pairwise_rows) if frame_type != 'point' else {}
            point_scores = _aggregate_inter_model_point_scores(pairwise_rows, float(build_result.options.point_match_radius)) if frame_type == 'point' else {}

            export_priority = float(_clip01(1.0 - labeled_best)) if labeled_best is not None else float(_clip01(disagreement))
            metric_values = {
                'overall_frame_score': export_priority,
                'export_priority_score': export_priority,
                'model_model_score': model_model_score,
                'disagreement_score': disagreement,
            }
            if model_labeled_score is not None:
                metric_values['model_labeled_score'] = float(model_labeled_score)
            if labeled_best is not None:
                metric_values['labeled_best_quality'] = labeled_best
            if labeled_mean is not None:
                metric_values['labeled_mean_quality'] = labeled_mean
            metric_values.update(polygon_scores)
            metric_values.update(point_scores)
            for model_id, confidence_row in model_confidence.items():
                if hasattr(confidence_row, 'mean_object_confidence'):
                    metric_values[_model_metric_key('model_confidence', str(model_id))] = float(getattr(confidence_row, 'frame_uncertainty_score', 0.0))
                    metric_values[_model_metric_key('model_uncertain_fraction', str(model_id))] = float(getattr(confidence_row, 'uncertain_fraction', 0.0))
                elif hasattr(confidence_row, 'mean_point_confidence'):
                    metric_values[_model_metric_key('model_confidence', str(model_id))] = float(getattr(confidence_row, 'frame_uncertainty_score', 0.0))
                    metric_values[_model_metric_key('model_point_contrast', str(model_id))] = float(getattr(confidence_row, 'mean_point_contrast', 0.0))

            summary = FrameAnalysisSummary(
                is_labeled=labeled_best is not None,
                disagreement_score=float(disagreement),
                temporal_instability=0.0,
                structural_anomaly=0.0,
                labeled_best_quality=labeled_best,
                labeled_mean_quality=labeled_mean,
                export_priority_score=float(export_priority),
                metric_values=metric_values,
                model_metrics=model_metrics,
                model_confidence=model_confidence,
                model_diagnostics=model_diagnostics,
                pairwise_metrics=pairwise_rows,
                notes=(('labeled' if labeled_best is not None else 'unlabeled'),),
                frame_type=frame_type,
            )
            updated_records_by_key[str(record.key)] = replace(record, summary=summary)
    finally:
        _clear_runtime_image_caches()

    updated_records = [updated_records_by_key[str(record.key)] for record in records if str(record.key) in updated_records_by_key]
    metric_is_required = _metric_requires_model_confidence(active_metric)
    metric_values_by_key: dict[str, float] = {}
    for record in updated_records:
        value = _record_metric_value(record.summary, active_metric)
        if value is None and metric_is_required:
            continue
        metric_values_by_key[str(record.key)] = float(value or 0.0)
    absolute_scores = list(metric_values_by_key.values())
    min_absolute = min(absolute_scores) if absolute_scores else 0.0
    max_absolute = max(absolute_scores) if absolute_scores else 0.0
    span = max(EPS, max_absolute - min_absolute)
    higher_is_better = metric_higher_is_better(active_metric)

    scored_records: list[FrameRecord] = []
    best_key = None
    best_value = None
    for record in updated_records:
        absolute = metric_values_by_key.get(str(record.key))
        if absolute is None:
            scored_records.append(replace(
                record,
                score=0.0,
                absolute_score=None,
                relative_score=None,
                score_percentile=None,
                score_ready=False,
            ))
            continue
        relative = 0.0 if abs(max_absolute - min_absolute) <= EPS else (absolute - min_absolute) / span
        display = relative if higher_is_better else (1.0 - relative)
        scored = replace(record, score=float(display), absolute_score=float(absolute), relative_score=float(relative), score_ready=True)
        scored_records.append(scored)
        if best_value is None or (absolute > best_value if higher_is_better else absolute < best_value):
            best_value = absolute
            best_key = record.key

    percentile_map = compute_metric_percentiles(scored_records, active_metric)
    scored_records = [replace(record, score_percentile=float(percentile_map.get(record.key, 0.0))) for record in scored_records]
    model_ranking = _aggregate_model_ranking(scored_records, build_result.model_specs)
    available_metric_keys: list[str] = []
    seen_metric_keys: set[str] = set()
    for metric_key_candidate in _available_metric_keys_for_models(build_result.model_specs):
        if metric_key_candidate not in seen_metric_keys:
            available_metric_keys.append(metric_key_candidate)
            seen_metric_keys.add(metric_key_candidate)
    for record in scored_records:
        summary = record.summary
        if summary is None:
            continue
        for metric_key_candidate in summary.metric_values.keys():
            metric_key_str = str(metric_key_candidate)
            if metric_key_str not in seen_metric_keys:
                available_metric_keys.append(metric_key_str)
                seen_metric_keys.add(metric_key_str)
    return replace(build_result, records=tuple(scored_records), min_score=min((record.score for record in scored_records), default=0.0), max_score=max((record.score for record in scored_records), default=0.0), scores_computed=True, best_match_key=best_key, min_absolute_score=min_absolute, max_absolute_score=max_absolute, selected_metric_key=active_metric, model_ranking=model_ranking, available_metric_keys=tuple(available_metric_keys))
