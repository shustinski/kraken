"""repository: build model payloads."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    FrameRecord,
    GeometryMode,
    ModelDiagnosticMetrics,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import _clip01

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.build_prediction_view_gray import build_prediction_view_from_gray

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.internal_confidence_probability_map import (
    _mask_from_gray,
    _prob_from_gray,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.labeled_quality import _labeled_quality

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_structure import _mask_structure

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.pairwise_model_comparisons import (
    _infer_geometry_mode,
    _point_diagnostic_metrics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.point_internal_confidence import _point_internal_confidence

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.point_labeled_quality import _point_labeled_quality

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_frame_confidence import _polygon_frame_confidence

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.record_payload_cache_key import _load_optional_gray

def _build_model_payloads(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    *,
    analysis_max_side: int | None = None,
    geometry_mode: GeometryMode = GeometryMode.MASK,
    point_match_radius: float = 3.0,
    boundary_radius: int = 1,
    confidence_uncertainty_delta: float = MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    point_confidence_radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    polygon_confidence_summary: str = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    include_confidence_objects: bool = True,
    include_original_gray: bool = True,
    include_model_confidence: bool = True,
    include_pairwise_metrics: bool = True,
    include_model_metrics: bool = True,
    include_model_diagnostics: bool = True,
    include_structure_details: bool = True,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, bool],
    np.ndarray | None,
    np.ndarray | None,
    dict[str, object] | None,
    GeometryMode,
    dict[str, object],
    object | None,
]:
    include_pairwise_metrics = bool(include_pairwise_metrics)
    include_model_metrics = bool(include_model_metrics)
    include_model_diagnostics = bool(include_model_diagnostics)
    include_model_confidence = bool(include_model_confidence)
    include_mask_structures = include_pairwise_metrics or include_model_metrics or include_model_diagnostics
    include_boundary_distance = include_pairwise_metrics or include_model_metrics

    original_gray = _load_optional_gray(record.original_path, max_side=analysis_max_side) if include_original_gray else None
    gt_gray = _load_optional_gray(record.gt_path, max_side=analysis_max_side) if include_model_metrics else None
    gt_mask = _mask_from_gray(gt_gray) if gt_gray is not None else None
    gt_structure = _mask_structure(
        gt_mask,
        include_skeleton=include_structure_details,
        include_boundary_distance=include_boundary_distance,
    ) if gt_mask is not None else None

    probabilities: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    model_structures: dict[str, dict[str, object]] = {}
    model_diagnostics: dict[str, object] = {}
    model_metrics: dict[str, object] = {}
    model_confidence: dict[str, object] = {}
    model_confidence_output_available: dict[str, bool] = {}
    model_views: dict[str, object] = {}

    target_shape = tuple(int(v) for v in gt_gray.shape) if gt_gray is not None else None
    if target_shape is None and original_gray is not None:
        target_shape = tuple(int(v) for v in original_gray.shape)

    loaded_rows: list[tuple[ModelSpec, np.ndarray, np.ndarray, bool]] = []
    first_point_candidate = None
    for spec in model_specs:
        mask_gray = _load_optional_gray(record.model_mask_paths.get(spec.model_id), target_shape=target_shape, max_side=analysis_max_side)
        if mask_gray is None:
            continue
        if target_shape is None:
            target_shape = tuple(int(v) for v in mask_gray.shape)
        if original_gray is not None and tuple(int(v) for v in original_gray.shape) != target_shape:
            original_gray = _load_optional_gray(record.original_path, target_shape=target_shape, max_side=analysis_max_side)
        if gt_gray is not None and tuple(int(v) for v in gt_gray.shape) != target_shape:
            gt_gray = _load_optional_gray(record.gt_path, target_shape=target_shape, max_side=analysis_max_side)
            gt_mask = _mask_from_gray(gt_gray) if gt_gray is not None else None
            gt_structure = _mask_structure(
                gt_mask,
                include_skeleton=include_structure_details,
                include_boundary_distance=include_boundary_distance,
            ) if gt_mask is not None else None
        prob_gray = _load_optional_gray(record.model_prob_paths.get(spec.model_id), target_shape=target_shape, max_side=analysis_max_side)
        uses_binary_probability_proxy = prob_gray is None
        model_confidence_output_available[spec.model_id] = not uses_binary_probability_proxy
        if prob_gray is None:
            prob_gray = mask_gray
        loaded_rows.append((spec, mask_gray, prob_gray, uses_binary_probability_proxy))
        if geometry_mode == GeometryMode.AUTO and first_point_candidate is None:
            first_point_candidate = build_prediction_view_from_gray(spec.display_name, np.asarray(prob_gray, dtype=np.uint8), threshold=int(round(float(spec.threshold) * 255.0)))

    resolved_geometry_mode = geometry_mode
    if resolved_geometry_mode == GeometryMode.AUTO:
        resolved_geometry_mode = _infer_geometry_mode(first_point_candidate) if first_point_candidate is not None else GeometryMode.MASK

    gt_point_view = None
    if resolved_geometry_mode == GeometryMode.POINT and gt_gray is not None:
        gt_point_view = build_prediction_view_from_gray('ground_truth', np.asarray(gt_gray, dtype=np.uint8), threshold=128)

    for spec, mask_gray, prob_gray, uses_binary_probability_proxy in loaded_rows:
        prob_map = _prob_from_gray(prob_gray)
        mask = _mask_from_gray(mask_gray, threshold=spec.threshold)
        probabilities[spec.model_id] = prob_map.astype(np.float32)
        masks[spec.model_id] = mask.astype(bool)

        if resolved_geometry_mode == GeometryMode.POINT:
            need_prediction_view = include_pairwise_metrics or include_model_metrics or include_model_confidence or include_model_diagnostics
            prediction_view = None
            if need_prediction_view:
                prediction_view = build_prediction_view_from_gray(spec.display_name, np.asarray(prob_gray, dtype=np.uint8), threshold=int(round(float(spec.threshold) * 255.0)))
            if prediction_view is not None and (include_pairwise_metrics or include_model_metrics):
                model_views[spec.model_id] = prediction_view
            diagnostics = _point_diagnostic_metrics(prediction_view) if (prediction_view is not None and include_model_diagnostics) else None
            if diagnostics is not None:
                model_diagnostics[spec.model_id] = diagnostics
            if include_model_confidence and prediction_view is not None:
                model_confidence[spec.model_id] = _point_internal_confidence(
                    prediction_view,
                    neighborhood_radius=int(point_confidence_radius),
                    include_objects=include_confidence_objects,
                )
            if prediction_view is not None and gt_point_view is not None and include_model_metrics:
                model_metrics[spec.model_id] = _point_labeled_quality(prediction_view, gt_point_view, point_match_radius)
            if diagnostics is not None and prediction_view is not None:
                model_structures[spec.model_id] = {
                    'component_count': int(diagnostics.point_count),
                    'area_fraction': float(max(0.0, min(1.0, diagnostics.point_count / max(1.0, float(prediction_view.pred_gray.size))))),
                    'skeleton_length': float(diagnostics.mean_radius),
                }
            continue

        mask_structure = None
        if include_mask_structures:
            mask_structure = _mask_structure(
                mask,
                include_skeleton=include_structure_details,
                include_boundary_distance=include_boundary_distance,
            )
            model_structures[spec.model_id] = mask_structure
        if include_model_diagnostics and mask_structure is not None:
            area_fraction = float(mask_structure['area_fraction'])
            proxy_score = float(_clip01(1.0 - area_fraction))
            model_diagnostics[spec.model_id] = ModelDiagnosticMetrics(
                area_fraction=area_fraction,
                component_count=int(mask_structure['component_count']),
                skeleton_length=float(mask_structure['skeleton_length']),
                proxy_score=proxy_score,
            )
        if include_model_confidence:
            model_confidence[spec.model_id] = _polygon_frame_confidence(
                prob_map,
                mask,
                uncertainty_delta=float(confidence_uncertainty_delta),
                summary_metric=str(polygon_confidence_summary),
                allow_binary_proxy=bool(uses_binary_probability_proxy),
            )
        if include_model_metrics and gt_mask is not None and gt_structure is not None:
            if mask_structure is None:
                mask_structure = _mask_structure(
                    mask,
                    include_skeleton=include_structure_details,
                    include_boundary_distance=True,
                )
                model_structures[spec.model_id] = mask_structure
            model_metrics[spec.model_id] = _labeled_quality(
                prob_map,
                mask,
                gt_mask,
                pred_structure=mask_structure,
                gt_structure=gt_structure,
                boundary_radius=boundary_radius,
            )

    return (
        probabilities,
        masks,
        model_structures,
        model_diagnostics,
        model_metrics,
        model_confidence,
        model_confidence_output_available,
        original_gray,
        gt_mask,
        gt_structure,
        resolved_geometry_mode,
        model_views,
        gt_point_view,
    )
