"""repository: load frame detail base."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildResult,
    FolderSpec,
    FrameRecord,
    GeometryMode,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.build_model_payloads import _build_model_payloads

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.build_prediction_view_gray import build_prediction_view_from_gray

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.extract_original_frame_features import extract_original_frame_features

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.internal_confidence_probability_map import (
    _detail_base_payload_cache_key,
    _load_cached_detail_payload,
    _mask_from_gray,
    _store_cached_detail_payload,
    _with_selected_detail_payload,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.pairwise_model_comparisons import _consensus_probability

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.record_payload_cache_key import _load_optional_gray

def load_frame_detail_base(
    record: FrameRecord,
    build_result: BuildResult,
    model_id: str | None = None,
    *,
    max_side: int | None = None,
) -> dict[str, object]:
    active_record = record
    active_build_result = build_result
    if not active_build_result.model_specs:
        first_path = str(active_record.first_path or "")
        second_path = str(active_record.second_path or "")
        if not first_path and active_record.model_mask_paths:
            first_path = str(next(iter(active_record.model_mask_paths.values()), ""))
        if not second_path and len(active_record.model_mask_paths) > 1:
            second_path = str(list(active_record.model_mask_paths.values())[1])
        if not second_path:
            second_path = first_path
        if first_path:
            compat_specs = (
                ModelSpec(model_id="first", display_name="1", mask_folder=Path(first_path).parent, threshold=0.5),
                ModelSpec(model_id="second", display_name="2", mask_folder=Path(second_path).parent if second_path else Path(first_path).parent, threshold=0.5),
            )
            active_record = replace(
                active_record,
                model_mask_paths={"first": first_path, "second": second_path},
                model_prob_paths={"first": "", "second": ""},
                original_path=active_record.original_path or active_record.base_path,
                base_path=active_record.base_path or active_record.original_path,
            )
            active_build_result = replace(
                active_build_result,
                model_specs=compat_specs,
                original_folder=active_build_result.original_folder or active_build_result.base_folder,
                first_folder=active_build_result.first_folder or FolderSpec(path=compat_specs[0].mask_folder, label=compat_specs[0].display_name),
                second_folder=active_build_result.second_folder or FolderSpec(path=compat_specs[1].mask_folder, label=compat_specs[1].display_name),
                base_folder=active_build_result.base_folder or active_build_result.original_folder,
                options=replace(active_build_result.options, geometry_mode=GeometryMode.MASK),
            )

    target_model_id = model_id or (active_build_result.model_specs[0].model_id if active_build_result.model_specs else None)
    cache_key = _detail_base_payload_cache_key(active_record, active_build_result, max_side)
    cached = _load_cached_detail_payload(cache_key)
    if isinstance(cached, dict):
        return _with_selected_detail_payload(cached, target_model_id)

    boundary_radius = int(getattr(active_build_result.options, 'boundary_radius', 1) or 1)
    confidence_uncertainty_delta = float(getattr(active_build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA))
    point_confidence_radius = int(getattr(active_build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS)
    polygon_confidence_summary = str(getattr(active_build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)
    summary_pairwise = tuple(active_record.summary.pairwise_metrics) if active_record.summary is not None and active_record.summary.pairwise_metrics else ()
    summary_model_metrics = dict(getattr(active_record.summary, 'model_metrics', {}) or {}) if active_record.summary is not None else {}
    include_pairwise_metrics = False
    include_model_metrics = False
    include_model_diagnostics = bool(active_build_result.options.geometry_mode in {GeometryMode.POINT, GeometryMode.AUTO})

    probabilities, masks, _model_structures, model_diagnostics, model_metrics, model_confidence, model_confidence_output_available, original_gray, gt_mask, _gt_structure, detail_geometry_mode, model_views, gt_point_view = _build_model_payloads(
        active_record,
        active_build_result.model_specs,
        analysis_max_side=max_side,
        geometry_mode=active_build_result.options.geometry_mode,
        point_match_radius=float(active_build_result.options.point_match_radius),
        boundary_radius=boundary_radius,
        confidence_uncertainty_delta=confidence_uncertainty_delta,
        point_confidence_radius=point_confidence_radius,
        polygon_confidence_summary=polygon_confidence_summary,
        include_confidence_objects=False,
        include_model_confidence=False,
        include_pairwise_metrics=include_pairwise_metrics,
        include_model_metrics=include_model_metrics,
        include_model_diagnostics=include_model_diagnostics,
        include_structure_details=True,
    )
    if summary_model_metrics:
        model_metrics = {**summary_model_metrics, **model_metrics}

    gt_gray = _load_optional_gray(active_record.gt_path, target_shape=tuple(int(v) for v in gt_mask.shape) if gt_mask is not None else (tuple(int(v) for v in original_gray.shape) if original_gray is not None else None), max_side=max_side)
    if gt_gray is not None:
        gt_mask = _mask_from_gray(gt_gray)
        if detail_geometry_mode == GeometryMode.POINT:
            gt_point_view = build_prediction_view_from_gray('ground_truth', np.asarray(gt_gray, dtype=np.uint8), threshold=128)
    model_display_names = {spec.model_id: spec.display_name for spec in active_build_result.model_specs}
    selected_model_id = target_model_id if target_model_id in probabilities else (next(iter(probabilities.keys()), None))
    summary_confidence = {}
    if active_record.summary is not None and getattr(active_record.summary, 'model_confidence', None):
        summary_confidence = dict(getattr(active_record.summary, 'model_confidence', {}) or {})
    if summary_confidence:
        model_confidence = {**summary_confidence, **model_confidence}
    fallback_prob = np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
    selected_prob = probabilities.get(selected_model_id, fallback_prob)
    selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
    consensus_prob = _consensus_probability(list(probabilities.values())) if probabilities else selected_prob
    consensus_mask = np.asarray(consensus_prob >= 0.5, dtype=bool)

    detail_payload = {
        "model_ids": tuple(probabilities.keys()),
        "model_display_names": model_display_names,
        "selected_model_id": selected_model_id,
        "original_gray": original_gray,
        "gt_gray": gt_gray,
        "gt_mask": gt_mask,
        "selected_prob": np.asarray(selected_prob, dtype=np.float32),
        "selected_mask": np.asarray(selected_mask, dtype=bool),
        "model_probabilities": probabilities,
        "model_masks": masks,
        "model_views": model_views,
        "gt_point_view": gt_point_view,
        "consensus_prob": consensus_prob,
        "consensus_mask": consensus_mask,
        "pairwise_model_comparisons": summary_pairwise,
        "frame_metrics": dict(active_record.summary.metric_values) if active_record.summary is not None else {},
        "model_metrics": model_metrics,
        "model_confidence": model_confidence,
        "model_confidence_output_available": model_confidence_output_available,
        "model_diagnostics": model_diagnostics,
        "geometry_mode": detail_geometry_mode.value,
        "point_match_radius": float(active_build_result.options.point_match_radius),
        "boundary_radius": boundary_radius,
        "confidence_uncertainty_delta": confidence_uncertainty_delta,
        "point_confidence_radius": point_confidence_radius,
        "polygon_confidence_summary": polygon_confidence_summary,
        "original_features": extract_original_frame_features(original_gray),
    }
    _store_cached_detail_payload(cache_key, detail_payload)
    return _with_selected_detail_payload(detail_payload, target_model_id)
