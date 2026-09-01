"""Metric identifiers, lookup, direction, ranking, and percentile helpers."""

from __future__ import annotations

from .repository_shared import (
    BuildResult,
    CONFIDENCE_PAIR_METRIC_OPERATIONS,
    ComparisonPairSelection,
    EXPORT_SELECTION_MODE_COUNT,
    EXPORT_SELECTION_MODE_PERCENT,
    EXPORT_SELECTION_MODE_PERCENTILE,
    FrameAnalysisSummary,
    FrameRecord,
    ModelSpec,
    PAIR_METRIC_OPERATIONS,
    math,
    np,
)


def _normalized_pair_operations(operations: tuple[str, ...] | list[str] | set[str] | None) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for operation in operations or ():
        key = str(operation or "").strip().lower()
        if key not in PAIR_METRIC_OPERATIONS or key in seen:
            continue
        ordered.append(key)
        seen.add(key)
    return tuple(ordered)


def _normalized_comparison_pairs(
    pairs: tuple[ComparisonPairSelection, ...] | list[ComparisonPairSelection] | None,
) -> tuple[ComparisonPairSelection, ...]:
    normalized: list[ComparisonPairSelection] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs or ():
        model_a = str(getattr(pair, "model_a_id", "") or "").strip()
        model_b = str(getattr(pair, "model_b_id", "") or "").strip()
        if not model_a or not model_b or model_a == model_b:
            continue
        operations = _normalized_pair_operations(getattr(pair, "operations", ()))
        if not operations:
            continue
        key = (model_a, model_b)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(ComparisonPairSelection(model_a, model_b, operations))
    return tuple(normalized)


def pair_metric_key(model_a_id: str, model_b_id: str, operation: str) -> str:
    return f"pair::{model_a_id}::{model_b_id}::{operation}"


def confidence_pair_metric_key(model_a_id: str, model_b_id: str, operation: str) -> str:
    return f"confidence_pair::{model_a_id}::{model_b_id}::{operation}"


def combined_pair_metric_key(model_a_id: str, model_b_id: str, operation: str = "risk") -> str:
    return f"combined_pair::{model_a_id}::{model_b_id}::{operation}"


def parse_pair_metric_key(metric_key: str | None) -> tuple[str, str, str] | None:
    parts = str(metric_key or "").split("::")
    if len(parts) != 4 or parts[0] != "pair":
        return None
    model_a, model_b, operation = (str(parts[1]), str(parts[2]), str(parts[3]).lower())
    if not model_a or not model_b or operation not in PAIR_METRIC_OPERATIONS:
        return None
    return model_a, model_b, operation


def parse_confidence_pair_metric_key(metric_key: str | None) -> tuple[str, str, str] | None:
    parts = str(metric_key or "").split("::")
    if len(parts) != 4 or parts[0] != "confidence_pair":
        return None
    model_a, model_b, operation = (str(parts[1]), str(parts[2]), str(parts[3]).lower())
    if not model_a or not model_b or operation not in CONFIDENCE_PAIR_METRIC_OPERATIONS:
        return None
    return model_a, model_b, operation


def parse_combined_pair_metric_key(metric_key: str | None) -> tuple[str, str, str] | None:
    parts = str(metric_key or "").split("::")
    if len(parts) != 4 or parts[0] != "combined_pair":
        return None
    model_a, model_b, operation = (str(parts[1]), str(parts[2]), str(parts[3]).lower())
    if not model_a or not model_b or operation != "risk":
        return None
    return model_a, model_b, operation


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
    if "::" not in str(metric_key):
        return None
    family, model_id = str(metric_key).split("::", 1)
    return (family, model_id) if family and model_id else None


def _available_metric_keys_for_models(
    model_specs: tuple[ModelSpec, ...], records: tuple[FrameRecord, ...] | list[FrameRecord] | None = None
) -> tuple[str, ...]:
    if len(model_specs) == 1 and str(getattr(model_specs[0], "model_id", "") or "") == "base_layer":
        return ("overall_frame_score",)
    keys = [
        "overall_frame_score",
        "export_priority_score",
        "model_model_score",
        "disagreement_score",
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
    model_ids_with_output = None
    if records is not None:
        model_ids_with_output = {
            str(model_id)
            for record in records
            for model_id, path_text in (record.model_prob_paths or {}).items()
            if bool(path_text)
        }
    if model_ids_with_output is None:
        confidence_model_count = sum(1 for spec in model_specs if spec.prob_folder is not None)
    else:
        confidence_model_count = sum(
            1 for spec in model_specs if spec.prob_folder is not None and str(spec.model_id) in model_ids_with_output
        )
    if confidence_model_count >= 2:
        keys.extend(
            [
                "confidence_model_score",
                "confidence_difference_score",
                "confidence_bce_score",
                "confidence_threshold_crossing_score",
            ]
        )
    for spec in model_specs:
        keys.append(_model_metric_key("model_confidence", spec.model_id))
        keys.append(_model_metric_key("model_uncertain_fraction", spec.model_id))
        keys.append(_model_metric_key("model_point_contrast", spec.model_id))
        if spec.prob_folder is not None and (
            model_ids_with_output is None or str(spec.model_id) in model_ids_with_output
        ):
            keys.append(_model_metric_key("model_output_confidence", spec.model_id))
    return tuple(keys)


def metric_value_for_record(record: FrameRecord, metric_key: str) -> float | None:
    return _record_metric_value(record.summary, metric_key)


def metric_higher_is_better(metric_key: str) -> bool:
    metric_key = str(metric_key or "")
    parsed_pair = parse_pair_metric_key(metric_key)
    if parsed_pair is not None:
        _model_a, _model_b, operation = parsed_pair
        return operation in {"iou", "dice"}
    parsed_confidence_pair = parse_confidence_pair_metric_key(metric_key)
    if parsed_confidence_pair is not None:
        _model_a, _model_b, operation = parsed_confidence_pair
        return operation in {"correlation", "low_iou"}
    if parse_combined_pair_metric_key(metric_key) is not None:
        return False
    parsed = _parse_model_metric_key(metric_key)
    if parsed is not None:
        family, _model_id = parsed
        if family == "model_uncertain_fraction":
            return False
        if family == "model_confidence":
            return False
        if family == "model_output_confidence":
            return False
        if family == "model_point_contrast":
            return True
    if metric_key in {"bce", "grid_inspection_damage_score", "mean_localization_distance"}:
        return False
    if metric_key in {
        "model_model_score",
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
        "confidence_model_score",
        "confidence_difference_score",
        "confidence_bce_score",
        "confidence_threshold_crossing_score",
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


def rank_records_by_metric_badness(
    records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str
) -> list[FrameRecord]:
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


def compute_metric_percentiles(
    records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str
) -> dict[str, float]:
    ranked = rank_records_by_metric(records, metric_key)
    return _ranked_percentile_map(ranked)


def compute_metric_badness_percentiles(
    records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str
) -> dict[str, float]:
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
        selected = [
            record for record in ranked if float(percentiles.get(record.key, 0.0)) >= float(percentile_threshold)
        ]
        return tuple(selected or ranked[:1])
    count = max(1, int(top_k))
    return tuple(ranked[:count])
