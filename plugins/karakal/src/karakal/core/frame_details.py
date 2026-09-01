"""Lazy frame-detail payload construction."""

from __future__ import annotations

from .analysis_cache import (
    _detail_base_payload_cache_key,
    _detail_confidence_cache_key,
    _detail_confidence_payload_ready,
    _load_cached_detail_payload,
    _store_cached_detail_payload,
    _with_selected_detail_payload,
)

from .analytics import (
    _build_frame_comparison_result,
    _build_model_payloads,
    _comparison_metric_values,
)

from .confidence_analysis import (
    _point_internal_confidence,
    _polygon_frame_confidence,
)

from .image_io import (
    extract_original_frame_features,
)

from .mask_metrics import (
    _consensus_probability,
)

from .repository_shared import (
    BuildResult,
    FolderSpec,
    FrameComparisonResult,
    FrameRecord,
    GeometryMode,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    ModelSpec,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    Path,
    np,
    replace,
)


def _attach_detail_comparison_payload(
    payload: dict[str, object],
    record: FrameRecord,
    build_result: BuildResult,
) -> dict[str, object]:
    if isinstance(payload.get("comparison_result"), FrameComparisonResult):
        return payload
    try:
        masks = payload.get("model_masks") or {}
        probabilities = payload.get("model_output_probabilities") or payload.get("model_probabilities") or {}
        if not isinstance(masks, dict) or len(masks) < 2:
            return payload
        geometry_value = str(payload.get("geometry_mode") or build_result.options.geometry_mode.value)
        geometry_mode = GeometryMode.POINT if geometry_value == GeometryMode.POINT.value else GeometryMode.MASK
        comparison_result = _build_frame_comparison_result(
            frame_id=str(record.key),
            model_specs=tuple(build_result.model_specs),
            probabilities_by_model={
                str(key): np.asarray(value, dtype=np.float32) for key, value in probabilities.items()
            },
            masks_by_model={str(key): np.asarray(value, dtype=bool) for key, value in masks.items()},
            geometry_mode=geometry_mode,
            threshold=float(getattr(build_result.options, "mask_threshold", 0.5) or 0.5),
            consensus_threshold=0.5,
            connectivity=8,
            compute_level="fast",
        )
    except Exception:
        return payload
    if comparison_result is None:
        return payload
    payload["comparison_result"] = comparison_result
    payload["comparison_metrics"] = tuple(comparison_result.metrics)
    payload["comparison_events"] = tuple(comparison_result.events)
    payload["comparison_raster_layers"] = tuple(comparison_result.raster_layers)
    frame_metrics = dict(payload.get("frame_metrics") or {})
    frame_metrics.update(_comparison_metric_values(comparison_result))
    payload["frame_metrics"] = frame_metrics
    return payload


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
                ModelSpec(
                    model_id="second",
                    display_name="2",
                    mask_folder=Path(second_path).parent if second_path else Path(first_path).parent,
                    threshold=0.5,
                ),
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
                first_folder=active_build_result.first_folder
                or FolderSpec(path=compat_specs[0].mask_folder, label=compat_specs[0].display_name),
                second_folder=active_build_result.second_folder
                or FolderSpec(path=compat_specs[1].mask_folder, label=compat_specs[1].display_name),
                base_folder=active_build_result.base_folder or active_build_result.original_folder,
                options=replace(active_build_result.options, geometry_mode=GeometryMode.MASK),
            )

    target_model_id = model_id or (
        active_build_result.model_specs[0].model_id if active_build_result.model_specs else None
    )
    cache_key = _detail_base_payload_cache_key(active_record, active_build_result, max_side)
    cached = _load_cached_detail_payload(cache_key)
    if isinstance(cached, dict):
        cached = _attach_detail_comparison_payload(cached, active_record, active_build_result)
        return _with_selected_detail_payload(cached, target_model_id)

    boundary_radius = int(getattr(active_build_result.options, "boundary_radius", 1) or 1)
    confidence_uncertainty_delta = float(
        getattr(active_build_result.options, "confidence_uncertainty_delta", MODEL_CONFIDENCE_UNCERTAIN_DELTA)
    )
    point_confidence_radius = int(
        getattr(active_build_result.options, "point_confidence_radius", POINT_CONFIDENCE_NEIGHBOR_RADIUS)
        or POINT_CONFIDENCE_NEIGHBOR_RADIUS
    )
    polygon_confidence_summary = str(
        getattr(active_build_result.options, "polygon_confidence_summary", POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)
        or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
    )
    summary_pairwise = (
        tuple(active_record.summary.pairwise_metrics)
        if active_record.summary is not None and active_record.summary.pairwise_metrics
        else ()
    )
    include_pairwise_metrics = False
    include_model_diagnostics = bool(
        active_build_result.options.geometry_mode in {GeometryMode.POINT, GeometryMode.AUTO}
    )

    (
        probabilities,
        output_probabilities,
        masks,
        model_source_grays,
        _model_structures,
        model_diagnostics,
        model_confidence,
        model_confidence_output_available,
        original_gray,
        detail_geometry_mode,
        model_views,
    ) = _build_model_payloads(
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
        include_model_diagnostics=include_model_diagnostics,
        include_structure_details=True,
    )
    model_display_names = {spec.model_id: spec.display_name for spec in active_build_result.model_specs}
    model_output_display_names = {
        spec.model_id: str(
            spec.prob_folder.name if spec.prob_folder is not None and spec.prob_folder.name else spec.display_name
        )
        for spec in active_build_result.model_specs
    }
    selected_model_id = (
        target_model_id if target_model_id in probabilities else (next(iter(probabilities.keys()), None))
    )
    summary_confidence = {}
    if active_record.summary is not None and getattr(active_record.summary, "model_confidence", None):
        summary_confidence = dict(getattr(active_record.summary, "model_confidence", {}) or {})
    if summary_confidence:
        model_confidence = {**summary_confidence, **model_confidence}
    fallback_prob = (
        np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
    )
    selected_prob = probabilities.get(selected_model_id, fallback_prob)
    selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
    consensus_prob = _consensus_probability(list(probabilities.values())) if probabilities else selected_prob
    consensus_mask = np.asarray(consensus_prob >= 0.5, dtype=bool)

    detail_payload = {
        "model_ids": tuple(probabilities.keys()),
        "model_display_names": model_display_names,
        "model_output_display_names": model_output_display_names,
        "selected_model_id": selected_model_id,
        "original_gray": original_gray,
        "selected_prob": np.asarray(selected_prob, dtype=np.float32),
        "selected_mask": np.asarray(selected_mask, dtype=bool),
        "model_probabilities": probabilities,
        "model_output_probabilities": output_probabilities,
        "model_masks": masks,
        "model_source_grays": model_source_grays,
        "model_views": model_views,
        "consensus_prob": consensus_prob,
        "consensus_mask": consensus_mask,
        "pairwise_model_comparisons": summary_pairwise,
        "frame_metrics": dict(active_record.summary.metric_values) if active_record.summary is not None else {},
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
    detail_payload = _attach_detail_comparison_payload(detail_payload, active_record, active_build_result)
    _store_cached_detail_payload(cache_key, detail_payload)
    return _with_selected_detail_payload(detail_payload, target_model_id)


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
        target_model_id = (
            "first"
            if (record.first_path or "first" in record.model_mask_paths)
            else next(iter(record.model_mask_paths.keys()), None)
        )
    if not target_model_id:
        return None

    payload = (
        detail_payload
        if detail_payload is not None
        else load_frame_detail_base(
            record,
            build_result,
            model_id=target_model_id,
            max_side=max_side,
        )
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
    confidence_uncertainty_delta = float(
        payload.get("confidence_uncertainty_delta")
        or float(getattr(build_result.options, "confidence_uncertainty_delta", MODEL_CONFIDENCE_UNCERTAIN_DELTA))
    )
    point_confidence_radius = int(
        payload.get("point_confidence_radius")
        or int(
            getattr(build_result.options, "point_confidence_radius", POINT_CONFIDENCE_NEIGHBOR_RADIUS)
            or POINT_CONFIDENCE_NEIGHBOR_RADIUS
        )
    )
    polygon_confidence_summary = str(
        payload.get("polygon_confidence_summary")
        or str(
            getattr(build_result.options, "polygon_confidence_summary", POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)
            or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
        )
    )

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
            allow_binary_proxy=True,
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
