"""repository: compute record payloads."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    FrameAnalysisSummary,
    FrameRecord,
    GeometryMode,
    ModelAggregateScore,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.iter_record_payloads import _iter_record_payloads

def _compute_record_payloads(
    records: list[FrameRecord],
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    max_workers: int,
    *,
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
    progress_callback=None,
    state_callback=None,
    cancel_check=None,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for record_key, payload in _iter_record_payloads(
        records,
        model_specs,
        analysis_max_side,
        max_workers,
        geometry_mode=geometry_mode,
        point_match_radius=point_match_radius,
        boundary_radius=boundary_radius,
        confidence_uncertainty_delta=confidence_uncertainty_delta,
        point_confidence_radius=point_confidence_radius,
        polygon_confidence_summary=polygon_confidence_summary,
        cache_enabled=cache_enabled,
        include_model_confidence=include_model_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
        include_model_metrics=include_model_metrics,
        progress_callback=progress_callback,
        state_callback=state_callback,
        cancel_check=cancel_check,
    ):
        if payload is not None:
            results[record_key] = payload
    return results

def _aggregate_model_ranking(records: list[FrameRecord], model_specs: tuple[ModelSpec, ...]) -> tuple[ModelAggregateScore, ...]:
    score_map: dict[str, list[float]] = {spec.model_id: [] for spec in model_specs}
    for record in records:
        summary = record.summary
        if summary is None:
            continue
        for model_id, metrics in (summary.model_metrics or {}).items():
            score = getattr(metrics, "quality_score", None)
            if score is None:
                continue
            score_map.setdefault(str(model_id), []).append(float(score))
    rows: list[ModelAggregateScore] = []
    for spec in model_specs:
        values = score_map.get(spec.model_id, [])
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        rows.append(ModelAggregateScore(
            model_id=spec.model_id,
            display_name=spec.display_name,
            labeled_frame_count=int(len(values)),
            mean_supervised_score=float(np.mean(array, dtype=np.float64)),
            median_supervised_score=float(np.median(array)),
            rank=0,
        ))
    rows.sort(key=lambda item: (item.mean_supervised_score, item.median_supervised_score, item.labeled_frame_count), reverse=True)
    return tuple(ModelAggregateScore(
        model_id=item.model_id,
        display_name=item.display_name,
        labeled_frame_count=item.labeled_frame_count,
        mean_supervised_score=item.mean_supervised_score,
        median_supervised_score=item.median_supervised_score,
        rank=index + 1,
    ) for index, item in enumerate(rows))

def _record_metric_value(summary: FrameAnalysisSummary | None, metric_key: str) -> float | None:
    if summary is None:
        return None
    value = summary.metric_values.get(metric_key)
    if value is not None:
        return float(value)
    if hasattr(summary, metric_key):
        raw = getattr(summary, metric_key)
        return None if raw is None else float(raw)
    return None

def _model_metric_key(metric_family: str, model_id: str) -> str:
    return f"{metric_family}::{model_id}"

def _parse_model_metric_key(metric_key: str) -> tuple[str, str] | None:
    if '::' not in str(metric_key):
        return None
    family, model_id = str(metric_key).split('::', 1)
    return (family, model_id) if family and model_id else None

def _available_metric_keys_for_models(model_specs: tuple[ModelSpec, ...]) -> tuple[str, ...]:
    keys = [
        "overall_frame_score",
        "export_priority_score",
        "model_model_score",
        "model_labeled_score",
        "disagreement_score",
        "labeled_best_quality",
        "labeled_mean_quality",
        "overall_polygon_score",
        "iou_score",
        "dice_score",
        "polygon_bce_score",
        "iou",
        "dice",
        "bce",
        "overall_point_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "localization_score",
        "precision",
        "recall",
        "f1",
        "mean_localization_distance",
    ]
    for spec in model_specs:
        keys.append(_model_metric_key("model_confidence", spec.model_id))
        keys.append(_model_metric_key("model_uncertain_fraction", spec.model_id))
        keys.append(_model_metric_key("model_point_contrast", spec.model_id))
    return tuple(keys)
