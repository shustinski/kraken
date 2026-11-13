"""Centralize analysis-mode and object-type routing for Karakal."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import BuildResult, GeometryMode

INTER_MODEL_ANALYSIS_MODE = "inter_model"
INTRA_MODEL_CONFIDENCE_MODE = "intra_model_confidence"
MODEL_OUTPUT_CONFIDENCE_MODE = "model_output_confidence"
CONFIDENCE_COMPARISON_MODE = "confidence_comparison"
SINGLE_RESULT_RISK_MODE = "single_result_risk"
HEURISTIC_CHECK_MODE = "heuristic_check"

SINGLE_RESULT_RISK_DISPLAY_KEYS: tuple[str, ...] = (
    "single_result_risk_score",
    "mask_structure_risk",
    "batch_outlier_risk",
    "source_alignment_risk",
    "source_mask_agreement_risk",
    "confidence_risk",
)
# The same keys as core/heuristic_check.py (HEURISTIC_CHECK_METRICS).
HEURISTIC_CHECK_DISPLAY_KEYS: tuple[str, ...] = (
    "heuristic_check_score",
    "heuristic_gross_risk",
    "heuristic_topology_risk",
    "heuristic_contamination_risk",
    "heuristic_width_risk",
    "heuristic_geometry_risk",
    "heuristic_batch_risk",
)
# Modes that rank one source by a 0-100 risk, explained by reasons.
RISK_ANALYSIS_MODES = frozenset({SINGLE_RESULT_RISK_MODE, HEURISTIC_CHECK_MODE})

POLYGON_OBJECT_TYPE = "polygon"
POINT_OBJECT_TYPE = "point"

ANALYSIS_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("analysis.mode.inter_model", INTER_MODEL_ANALYSIS_MODE),
    ("analysis.mode.confidence_comparison", CONFIDENCE_COMPARISON_MODE),
    ("analysis.mode.intra_model_confidence", INTRA_MODEL_CONFIDENCE_MODE),
    ("analysis.mode.model_output_confidence", MODEL_OUTPUT_CONFIDENCE_MODE),
    ("analysis.mode.single_result_risk", SINGLE_RESULT_RISK_MODE),
    ("analysis.mode.heuristic_check", HEURISTIC_CHECK_MODE),
)

OBJECT_TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("analysis.object_type.polygons", POLYGON_OBJECT_TYPE),
    ("analysis.object_type.points", POINT_OBJECT_TYPE),
)

INTER_MODEL_POLYGON_DISPLAY_KEYS: tuple[str, ...] = (
    "topology_risk",
    "photo_mismatch_risk",
    "photo_quality_risk",
    "overall_polygon_score",
    "iou_score",
    "dice_score",
    "polygon_bce_score",
)
INTER_MODEL_POLYGON_PERCENTILE_KEYS: tuple[str, ...] = (
    "topology_risk",
    "overall_polygon_score",
    "iou_score",
    "dice_score",
    "polygon_bce_score",
)

INTER_MODEL_POINT_DISPLAY_KEYS: tuple[str, ...] = (
    "topology_risk",
    "photo_mismatch_risk",
    "photo_quality_risk",
    "overall_point_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "localization_score",
)
INTER_MODEL_POINT_PERCENTILE_KEYS: tuple[str, ...] = (
    "topology_risk",
    "overall_point_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "localization_score",
)

SCORE_100_METRIC_KEYS = frozenset(
    {
        "topology_risk",
        "photo_mismatch_risk",
        "photo_quality_risk",
        "overall_polygon_score",
        "iou_score",
        "dice_score",
        "polygon_bce_score",
        "overall_point_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "localization_score",
        "confidence_model_score",
        "confidence_difference_score",
        "confidence_bce_score",
        "confidence_threshold_crossing_score",
        *SINGLE_RESULT_RISK_DISPLAY_KEYS,
        *HEURISTIC_CHECK_DISPLAY_KEYS,
    }
)

CONFIDENCE_COMPARISON_DISPLAY_KEYS: tuple[str, ...] = (
    "confidence_model_score",
    "confidence_difference_score",
    "confidence_bce_score",
    "confidence_threshold_crossing_score",
)

# Model-specific metric families (family::model) on the 0-100 scale, lower is better.
SCORE_100_LOWER_FAMILIES = frozenset({"photo_mismatch_risk", "confidence_ece", "confident_error_share"})

LOWER_IS_BETTER_METRIC_KEYS = frozenset(
    {
        "photo_mismatch_risk",
        "photo_quality_risk",
        "topology_risk",
        "topology_events",
        "bce",
        "grid_inspection_damage_score",
        "mean_localization_distance",
        *SINGLE_RESULT_RISK_DISPLAY_KEYS,
        *HEURISTIC_CHECK_DISPLAY_KEYS,
    }
)

CONFIDENCE_QUALITY_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.15, "score.level.low"),
    (0.35, "score.level.moderate"),
    (0.60, "score.level.elevated"),
)


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Describe the active analysis mode, object type, and confidence source."""

    analysis_mode: str
    object_type: str
    confidence_model_id: str | None = None


def normalize_analysis_mode(value: str | None) -> str:
    text = str(value or "")
    if text in {
        INTRA_MODEL_CONFIDENCE_MODE,
        MODEL_OUTPUT_CONFIDENCE_MODE,
        CONFIDENCE_COMPARISON_MODE,
        SINGLE_RESULT_RISK_MODE,
        HEURISTIC_CHECK_MODE,
    }:
        return text
    return INTER_MODEL_ANALYSIS_MODE


def normalize_object_type(value: str | None) -> str:
    return POINT_OBJECT_TYPE if str(value or "") == POINT_OBJECT_TYPE else POLYGON_OBJECT_TYPE


def object_type_from_geometry_mode(value: str | GeometryMode | None) -> str:
    if isinstance(value, GeometryMode):
        value = value.value
    return POINT_OBJECT_TYPE if str(value or "") == GeometryMode.POINT.value else POLYGON_OBJECT_TYPE


def geometry_mode_for_object_type(value: str | None) -> GeometryMode:
    return GeometryMode.POINT if normalize_object_type(value) == POINT_OBJECT_TYPE else GeometryMode.MASK


def confidence_metric_key(model_id: str) -> str:
    return f"model_confidence::{model_id}"


def model_output_confidence_metric_key(model_id: str) -> str:
    return f"model_output_confidence::{model_id}"


def confidence_metric_family(metric_key: str | None) -> tuple[str, str] | None:
    if "::" not in str(metric_key or ""):
        return None
    family, model_id = str(metric_key).split("::", 1)
    if not family or not model_id:
        return None
    return family, model_id


def confidence_quality_level_key(value: float | None) -> str | None:
    if value is None:
        return None
    numeric = float(value)
    if not (numeric >= 0.0 or numeric <= 0.0):
        return None
    for upper_bound, level_key in CONFIDENCE_QUALITY_THRESHOLDS:
        if numeric < float(upper_bound):
            return str(level_key)
    return "score.level.high"


def available_confidence_model_ids(build_result: BuildResult | None) -> tuple[str, ...]:
    if build_result is None:
        return tuple()
    return tuple(str(spec.model_id) for spec in build_result.model_specs)


def available_model_output_confidence_model_ids(build_result: BuildResult | None) -> tuple[str, ...]:
    if build_result is None:
        return tuple()
    model_ids: list[str] = []
    for spec in build_result.model_specs:
        if spec.prob_folder is None:
            continue
        model_id = str(spec.model_id)
        if any(bool((record.model_prob_paths or {}).get(model_id)) for record in build_result.records):
            model_ids.append(model_id)
    return tuple(model_ids)


def default_confidence_model_id(build_result: BuildResult | None, *, output_only: bool = False) -> str | None:
    model_ids = (
        available_model_output_confidence_model_ids(build_result)
        if output_only
        else available_confidence_model_ids(build_result)
    )
    return model_ids[0] if model_ids else None


def resolve_analysis_context(
    build_result: BuildResult | None,
    analysis_mode: str | None,
    object_type: str | None,
    *,
    confidence_model_id: str | None = None,
) -> AnalysisContext:
    normalized_mode = normalize_analysis_mode(analysis_mode)
    normalized_object_type = normalize_object_type(object_type)
    if normalized_mode in {INTRA_MODEL_CONFIDENCE_MODE, MODEL_OUTPUT_CONFIDENCE_MODE}:
        output_only = normalized_mode == MODEL_OUTPUT_CONFIDENCE_MODE
        valid_ids = set(
            available_model_output_confidence_model_ids(build_result)
            if output_only
            else available_confidence_model_ids(build_result)
        )
        resolved_model_id = (
            str(confidence_model_id)
            if confidence_model_id in valid_ids
            else default_confidence_model_id(build_result, output_only=output_only)
        )
        return AnalysisContext(normalized_mode, normalized_object_type, resolved_model_id)
    if normalized_mode == CONFIDENCE_COMPARISON_MODE:
        return AnalysisContext(normalized_mode, normalized_object_type, None)
    return AnalysisContext(normalized_mode, normalized_object_type, None)


def display_metric_keys(context: AnalysisContext) -> tuple[str, ...]:
    if context.analysis_mode == SINGLE_RESULT_RISK_MODE:
        return SINGLE_RESULT_RISK_DISPLAY_KEYS
    if context.analysis_mode == HEURISTIC_CHECK_MODE:
        return HEURISTIC_CHECK_DISPLAY_KEYS
    if context.analysis_mode == INTRA_MODEL_CONFIDENCE_MODE:
        if context.confidence_model_id is None:
            return tuple()
        return (confidence_metric_key(context.confidence_model_id),)
    if context.analysis_mode == MODEL_OUTPUT_CONFIDENCE_MODE:
        if context.confidence_model_id is None:
            return tuple()
        return (model_output_confidence_metric_key(context.confidence_model_id),)
    if context.analysis_mode == CONFIDENCE_COMPARISON_MODE:
        return CONFIDENCE_COMPARISON_DISPLAY_KEYS
    if context.object_type == POINT_OBJECT_TYPE:
        return INTER_MODEL_POINT_DISPLAY_KEYS
    return INTER_MODEL_POLYGON_DISPLAY_KEYS


def percentile_basis_keys(context: AnalysisContext) -> tuple[str, ...]:
    if context.analysis_mode == SINGLE_RESULT_RISK_MODE:
        return SINGLE_RESULT_RISK_DISPLAY_KEYS
    if context.analysis_mode == HEURISTIC_CHECK_MODE:
        return HEURISTIC_CHECK_DISPLAY_KEYS
    if context.analysis_mode == INTRA_MODEL_CONFIDENCE_MODE:
        if context.confidence_model_id is None:
            return tuple()
        return (confidence_metric_key(context.confidence_model_id),)
    if context.analysis_mode == MODEL_OUTPUT_CONFIDENCE_MODE:
        if context.confidence_model_id is None:
            return tuple()
        return (model_output_confidence_metric_key(context.confidence_model_id),)
    if context.analysis_mode == CONFIDENCE_COMPARISON_MODE:
        return CONFIDENCE_COMPARISON_DISPLAY_KEYS
    if context.object_type == POINT_OBJECT_TYPE:
        return INTER_MODEL_POINT_PERCENTILE_KEYS
    return INTER_MODEL_POLYGON_PERCENTILE_KEYS


def default_metric_key(context: AnalysisContext) -> str:
    keys = percentile_basis_keys(context)
    if keys:
        return keys[0]
    return "overall_frame_score"


# Direction, range, level and colour of every score live in core/score_scale.py (one scale for all
# modes); these names stay for the callers that import them from here.
from .score_scale import metric_is_lower_better as metric_is_lower_better  # noqa: E402
from .score_scale import metric_level_key as metric_level_key  # noqa: E402
from .score_scale import metric_uses_100_scale as metric_uses_100_scale  # noqa: E402
from .score_scale import metric_visual_ratio as metric_visual_ratio  # noqa: E402
