"""repository: metric higher better."""
from __future__ import annotations
import math
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    EXPORT_SELECTION_MODE_COUNT,
    EXPORT_SELECTION_MODE_PERCENT,
    EXPORT_SELECTION_MODE_PERCENTILE,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildOptions,
    BuildResult,
    ComparisonMode,
    FolderSpec,
    FrameRecord,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.collect_frame_records import collect_frame_records

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.compute_build_result_metrics import compute_build_result_metrics

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.compute_record_payloads import (
    _parse_model_metric_key,
    _record_metric_value,
)

def build_frame_records(
    first_folder: FolderSpec,
    second_folder: FolderSpec,
    options: BuildOptions,
    *,
    base_folder: FolderSpec | None = None,
    progress_callback=None,
    cancel_check=None,
) -> BuildResult:
    """Legacy-lite full build wrapper (index + metric compute)."""

    initial = collect_frame_records(first_folder, second_folder, options, base_folder=base_folder, cancel_check=cancel_check)
    return compute_build_result_metrics(
        initial,
        comparison_mode=options.comparison_mode,
        display_metric="relative",
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

def compute_build_result_mismatches(
    build_result: BuildResult,
    *,
    comparison_mode: ComparisonMode | None = None,
    display_metric: str = "relative",
    progress_callback=None,
    cancel_check=None,
) -> BuildResult:
    """Backward-compatible alias for the legacy lite API."""

    return compute_build_result_metrics(
        build_result,
        comparison_mode=comparison_mode,
        display_metric=display_metric,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

def metric_value_for_record(record: FrameRecord, metric_key: str) -> float | None:
    return _record_metric_value(record.summary, metric_key)

def metric_higher_is_better(metric_key: str) -> bool:
    metric_key = str(metric_key or "")
    parsed = _parse_model_metric_key(metric_key)
    if parsed is not None:
        family, _model_id = parsed
        if family == "model_uncertain_fraction":
            return False
        if family == "model_confidence":
            return False
        if family == "model_point_contrast":
            return True
    if metric_key in {"bce", "mean_localization_distance"}:
        return False
    if metric_key in {
        "labeled_best_quality",
        "labeled_mean_quality",
        "model_model_score",
        "model_labeled_score",
        "iou",
        "dice",
        "iou_score",
        "dice_score",
        "polygon_bce_score",
        "overall_polygon_score",
        "precision",
        "recall",
        "f1",
        "precision_score",
        "recall_score",
        "f1_score",
        "localization_score",
        "overall_point_score",
    }:
        return True
    return False

def metric_percentile_high_is_bad(metric_key: str) -> bool:
    """Compatibility helper for legacy callers.

    Score percentiles shown in the UI are goodness percentiles:
    higher percentile always means a better frame, regardless of metric sign.
    """

    return False

def rank_records_by_metric(records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str) -> list[FrameRecord]:
    higher_is_better = metric_higher_is_better(metric_key)
    scored: list[tuple[float, FrameRecord]] = []
    for record in records:
        value = metric_value_for_record(record, metric_key)
        if value is None:
            continue
        value_float = float(value)
        if not np.isfinite(value_float):
            continue
        scored.append((value_float, record))
    ranked = sorted(scored, key=lambda item: item[0], reverse=bool(higher_is_better))
    return [record for _value, record in ranked]

def rank_records_by_metric_badness(records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str) -> list[FrameRecord]:
    higher_is_better = metric_higher_is_better(metric_key)
    scored: list[tuple[float, FrameRecord]] = []
    for record in records:
        value = metric_value_for_record(record, metric_key)
        if value is None:
            continue
        value_float = float(value)
        if not np.isfinite(value_float):
            continue
        scored.append((value_float, record))
    ranked = sorted(scored, key=lambda item: item[0], reverse=not bool(higher_is_better))
    return [record for _value, record in ranked]

def _ranked_percentile_map(ranked: list[FrameRecord]) -> dict[str, float]:
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {ranked[0].key: 100.0}
    denominator = max(1, len(ranked) - 1)
    return {record.key: float(100.0 * (denominator - index) / denominator) for index, record in enumerate(ranked)}

def compute_metric_percentiles(records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str) -> dict[str, float]:
    ranked = rank_records_by_metric(records, metric_key)
    return _ranked_percentile_map(ranked)

def compute_metric_badness_percentiles(records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str) -> dict[str, float]:
    ranked = rank_records_by_metric_badness(records, metric_key)
    return _ranked_percentile_map(ranked)

def select_candidate_records(
    build_result: BuildResult,
    *,
    metric_key: str,
    selection_mode: str = EXPORT_SELECTION_MODE_COUNT,
    top_k: int = 32,
    top_percent: float = 10.0,
    percentile_threshold: float = 90.0,
) -> tuple[FrameRecord, ...]:
    ranked = rank_records_by_metric_badness(build_result.records, metric_key)
    if not ranked:
        return tuple()
    mode = str(selection_mode or EXPORT_SELECTION_MODE_COUNT)
    if mode == EXPORT_SELECTION_MODE_PERCENT:
        count = max(1, int(math.ceil(len(ranked) * max(0.0, float(top_percent)) / 100.0)))
        return tuple(ranked[:count])
    if mode == EXPORT_SELECTION_MODE_PERCENTILE:
        percentiles = compute_metric_badness_percentiles(ranked, metric_key)
        selected = [record for record in ranked if float(percentiles.get(record.key, 0.0)) >= float(percentile_threshold)]
        return tuple(selected or ranked[:1])
    count = max(1, int(top_k))
    return tuple(ranked[:count])
