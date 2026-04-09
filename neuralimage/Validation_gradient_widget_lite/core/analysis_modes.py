"""Centralize analysis-mode and object-type routing for the lite widget."""
from __future__ import annotations

from dataclasses import dataclass

from .domain import BuildResult, GeometryMode

INTER_MODEL_ANALYSIS_MODE = "inter_model"
INTRA_MODEL_CONFIDENCE_MODE = "intra_model_confidence"

POLYGON_OBJECT_TYPE = "polygon"
POINT_OBJECT_TYPE = "point"

ANALYSIS_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("analysis.mode.inter_model", INTER_MODEL_ANALYSIS_MODE),
    ("analysis.mode.intra_model_confidence", INTRA_MODEL_CONFIDENCE_MODE),
)

OBJECT_TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("analysis.object_type.polygons", POLYGON_OBJECT_TYPE),
    ("analysis.object_type.points", POINT_OBJECT_TYPE),
)

INTER_MODEL_POLYGON_DISPLAY_KEYS: tuple[str, ...] = (
    "overall_polygon_score",
    "iou_score",
    "dice_score",
    "polygon_bce_score",
    "iou",
    "dice",
    "bce",
)
INTER_MODEL_POLYGON_PERCENTILE_KEYS: tuple[str, ...] = (
    "overall_polygon_score",
    "iou_score",
    "dice_score",
    "polygon_bce_score",
)

INTER_MODEL_POINT_DISPLAY_KEYS: tuple[str, ...] = (
    "overall_point_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "localization_score",
    "precision",
    "recall",
    "f1",
    "mean_localization_distance",
)
INTER_MODEL_POINT_PERCENTILE_KEYS: tuple[str, ...] = (
    "overall_point_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "localization_score",
)

SCORE_100_METRIC_KEYS = frozenset({
    "overall_polygon_score",
    "iou_score",
    "dice_score",
    "polygon_bce_score",
    "overall_point_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "localization_score",
})

LOWER_IS_BETTER_METRIC_KEYS = frozenset({
    "bce",
    "mean_localization_distance",
})


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Describe the active analysis mode, object type, and confidence source."""

    analysis_mode: str
    object_type: str
    confidence_model_id: str | None = None


def normalize_analysis_mode(value: str | None) -> str:
    return INTRA_MODEL_CONFIDENCE_MODE if str(value or "") == INTRA_MODEL_CONFIDENCE_MODE else INTER_MODEL_ANALYSIS_MODE


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


def confidence_metric_family(metric_key: str | None) -> tuple[str, str] | None:
    if "::" not in str(metric_key or ""):
        return None
    family, model_id = str(metric_key).split("::", 1)
    if not family or not model_id:
        return None
    return family, model_id


def available_confidence_model_ids(build_result: BuildResult | None) -> tuple[str, ...]:
    if build_result is None:
        return tuple()
    return tuple(str(spec.model_id) for spec in build_result.model_specs)


def default_confidence_model_id(build_result: BuildResult | None) -> str | None:
    model_ids = available_confidence_model_ids(build_result)
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
    if normalized_mode == INTRA_MODEL_CONFIDENCE_MODE:
        valid_ids = set(available_confidence_model_ids(build_result))
        resolved_model_id = str(confidence_model_id) if confidence_model_id in valid_ids else default_confidence_model_id(build_result)
        return AnalysisContext(normalized_mode, normalized_object_type, resolved_model_id)
    return AnalysisContext(normalized_mode, normalized_object_type, None)


def display_metric_keys(context: AnalysisContext) -> tuple[str, ...]:
    if context.analysis_mode == INTRA_MODEL_CONFIDENCE_MODE:
        if context.confidence_model_id is None:
            return tuple()
        return (confidence_metric_key(context.confidence_model_id),)
    if context.object_type == POINT_OBJECT_TYPE:
        return INTER_MODEL_POINT_DISPLAY_KEYS
    return INTER_MODEL_POLYGON_DISPLAY_KEYS


def percentile_basis_keys(context: AnalysisContext) -> tuple[str, ...]:
    if context.analysis_mode == INTRA_MODEL_CONFIDENCE_MODE:
        if context.confidence_model_id is None:
            return tuple()
        return (confidence_metric_key(context.confidence_model_id),)
    if context.object_type == POINT_OBJECT_TYPE:
        return INTER_MODEL_POINT_PERCENTILE_KEYS
    return INTER_MODEL_POLYGON_PERCENTILE_KEYS


def default_metric_key(context: AnalysisContext) -> str:
    keys = percentile_basis_keys(context)
    if keys:
        return keys[0]
    return "overall_frame_score"


def metric_uses_100_scale(metric_key: str | None) -> bool:
    return str(metric_key or "") in SCORE_100_METRIC_KEYS


def metric_is_lower_better(metric_key: str | None) -> bool:
    return str(metric_key or "") in LOWER_IS_BETTER_METRIC_KEYS


def metric_visual_ratio(
    metric_key: str | None,
    value: float | None,
    *,
    point_match_radius: float,
    bce_score_cap: float,
) -> float | None:
    if value is None:
        return None
    metric = str(metric_key or "")
    numeric = float(value)
    if metric in SCORE_100_METRIC_KEYS:
        return max(0.0, min(numeric / 100.0, 1.0))
    if metric == "bce":
        return max(0.0, min(numeric / max(1e-9, float(bce_score_cap)), 1.0))
    if metric == "mean_localization_distance":
        return max(0.0, min(numeric / max(1e-9, float(point_match_radius)), 1.0))
    return max(0.0, min(numeric, 1.0))
