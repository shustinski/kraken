"""repository: load frame detail model."""
from __future__ import annotations
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildResult,
    FrameRecord,
    GeometryMode,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.internal_confidence_probability_map import (
    _detail_confidence_cache_key,
    _detail_confidence_payload_ready,
    _load_cached_detail_payload,
    _store_cached_detail_payload,
    _with_selected_detail_payload,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.load_frame_detail_base import load_frame_detail_base

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.point_internal_confidence import _point_internal_confidence

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_frame_confidence import _polygon_frame_confidence

def load_frame_detail_model_confidence(
    record: FrameRecord,
    build_result: BuildResult,
    model_id: str | None = None,
    *,
    max_side: int | None = None,
    detail_payload: dict[str, object] | None = None,
):
    target_model_id = model_id or (build_result.model_specs[0].model_id if build_result.model_specs else None)
    if target_model_id is None and (record.first_path or record.model_mask_paths):
        target_model_id = "first" if (record.first_path or "first" in record.model_mask_paths) else next(iter(record.model_mask_paths.keys()), None)
    if not target_model_id:
        return None

    payload = detail_payload if detail_payload is not None else load_frame_detail_base(
        record,
        build_result,
        model_id=target_model_id,
        max_side=max_side,
    )
    model_confidence = payload.setdefault("model_confidence", {})
    geometry_mode = str(payload.get("geometry_mode") or GeometryMode.MASK.value)
    existing = (model_confidence or {}).get(target_model_id)
    if _detail_confidence_payload_ready(existing, geometry_mode):
        return existing

    cache_key = _detail_confidence_cache_key(record, build_result, max_side, target_model_id)
    cached = _load_cached_detail_payload(cache_key)
    if _detail_confidence_payload_ready(cached, geometry_mode):
        model_confidence[target_model_id] = cached
        return cached

    probabilities = payload.get("model_probabilities") or {}
    masks = payload.get("model_masks") or {}
    boundary_radius = int(payload.get("boundary_radius") or int(getattr(build_result.options, 'boundary_radius', 1) or 1))
    confidence_uncertainty_delta = float(payload.get("confidence_uncertainty_delta") or float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)))
    point_confidence_radius = int(payload.get("point_confidence_radius") or int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS))
    polygon_confidence_summary = str(payload.get("polygon_confidence_summary") or str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED))

    if geometry_mode == GeometryMode.POINT.value:
        selected_view = (payload.get("model_views") or {}).get(target_model_id)
        if selected_view is None:
            return None
        confidence_row = _point_internal_confidence(
            selected_view,
            neighborhood_radius=point_confidence_radius,
            include_objects=True,
        )
    else:
        selected_probability = probabilities.get(target_model_id)
        selected_model_mask = masks.get(target_model_id)
        if selected_probability is None or selected_model_mask is None:
            return None
        confidence_row = _polygon_frame_confidence(
            selected_probability,
            selected_model_mask,
            uncertainty_delta=confidence_uncertainty_delta,
            summary_metric=polygon_confidence_summary,
            include_debug=True,
            allow_binary_proxy=not bool((record.model_prob_paths or {}).get(target_model_id)),
        )
    model_confidence[target_model_id] = confidence_row
    _store_cached_detail_payload(cache_key, confidence_row)
    return confidence_row

def load_frame_detail(
    record: FrameRecord,
    build_result: BuildResult,
    model_id: str | None = None,
    *,
    max_side: int | None = None,
    include_selected_confidence: bool = True,
) -> dict[str, object]:
    target_model_id = model_id or (build_result.model_specs[0].model_id if build_result.model_specs else None)
    detail_payload = load_frame_detail_base(
        record,
        build_result,
        model_id=target_model_id,
        max_side=max_side,
    )
    if include_selected_confidence and target_model_id is not None:
        confidence_row = load_frame_detail_model_confidence(
            record,
            build_result,
            model_id=target_model_id,
            max_side=max_side,
            detail_payload=detail_payload,
        )
        if confidence_row is not None:
            (detail_payload.setdefault("model_confidence", {}))[target_model_id] = confidence_row
    return _with_selected_detail_payload(detail_payload, target_model_id)
