"""Karakal result and diagnostic export workflows."""

from __future__ import annotations

from .analytics import (
    _build_frame_comparison_result,
    _build_model_payloads,
)

from .confidence_analysis import (
    _internal_confidence_probability_map,
    _mask_from_gray,
    _prob_from_gray,
)

from .image_io import (
    _grayscale_array_to_qimage,
    _load_export_grayscale_image,
    _rgb_array_to_qimage,
    _save_export_rgb_jpg,
    load_grayscale_image,
    natural_sort_key,
    resize_grayscale_image,
)

from .mask_metrics import (
    compute_comparison,
)

from .mask_primitives import (
    _boundary_mask,
    _distance_transform,
)

from .metric_keys import (
    compute_metric_percentiles,
    metric_value_for_record,
    select_candidate_records,
)

from .repository_shared import (
    BuildResult,
    ComparisonMode,
    EXPORT_MAX_WORKERS,
    EXPORT_SELECTION_MODE_COUNT,
    EXPORT_WORKER_ENV,
    FIRST_COMPLETED,
    FrameComparisonResult,
    FrameRecord,
    GridFrameAnalysisResult,
    INVALID_FILENAME_PATTERN,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    ModelSpec,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    Path,
    ThreadPoolExecutor,
    build_model_uncertainty,
    confidence_bad_area_intensity,
    detect_grid_cell_anomalies,
    json,
    math,
    np,
    os,
    replace,
    shutil,
    wait,
)


def _sequence_groups(records: tuple[FrameRecord, ...]) -> dict[str, list[FrameRecord]]:
    groups: dict[str, list[FrameRecord]] = {}
    for record in records:
        sequence_id = (
            record.identity.sequence_id if record.identity is not None and record.identity.sequence_id else "__root__"
        )
        groups.setdefault(sequence_id, []).append(record)
    for items in groups.values():
        items.sort(key=lambda item: natural_sort_key(item.key))
    return groups


def _safe_export_name(value: str | None, *, fallback: str) -> str:
    name = INVALID_FILENAME_PATTERN.sub("_", str(value or "").strip()).strip(" ._")
    return name or str(fallback)


def _unique_export_folder_name(base_name: str, used_names: set[str]) -> str:
    candidate = _safe_export_name(base_name, fallback="source")
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    index = 2
    while True:
        suffixed = f"{candidate}_{index}"
        if suffixed not in used_names:
            used_names.add(suffixed)
            return suffixed
        index += 1


def _rgb_tuple(
    value: tuple[int, int, int] | list[int] | np.ndarray | None, *, fallback: tuple[int, int, int]
) -> tuple[int, int, int]:
    if value is None:
        return fallback
    try:
        items = tuple(int(round(float(item))) for item in tuple(value)[:3])
    except Exception:
        return fallback
    if len(items) != 3:
        return fallback
    return tuple(max(0, min(255, item)) for item in items)


def _normalized_layer_unit(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3:
        values = values[..., 0]
    values_f = np.asarray(values, dtype=np.float32)
    if values_f.size <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    finite = np.isfinite(values_f)
    if not np.any(finite):
        return np.zeros_like(values_f, dtype=np.float32)
    result = np.nan_to_num(values_f, nan=0.0, posinf=1.0, neginf=0.0)
    finite_values = values_f[finite]
    max_value = float(np.max(finite_values))
    min_value = float(np.min(finite_values))
    if max_value > 1.0 and min_value >= 0.0 and max_value <= 255.0:
        result = result / 255.0
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def _colorize_layer_map(
    array: np.ndarray,
    *,
    map_color: tuple[int, int, int] | list[int] | np.ndarray | None = None,
    background_color: tuple[int, int, int] | list[int] | np.ndarray | None = None,
) -> np.ndarray:
    unit = _normalized_layer_unit(array)[..., None]
    foreground = np.asarray(_rgb_tuple(map_color, fallback=(255, 64, 64)), dtype=np.float32)
    background = np.asarray(_rgb_tuple(background_color, fallback=(128, 128, 128)), dtype=np.float32)
    rgb = background * (1.0 - unit) + foreground * unit
    return np.clip(np.round(rgb), 0.0, 255.0).astype(np.uint8)


def _model_display_name_map(build_result: BuildResult) -> dict[str, str]:
    return {
        str(spec.model_id): str(spec.display_name or spec.model_id) for spec in tuple(build_result.model_specs or ())
    }


def _comparison_result_for_export(record: FrameRecord, build_result: BuildResult) -> FrameComparisonResult | None:
    (
        probabilities,
        output_probabilities,
        masks,
        _source_grays,
        _model_structures,
        _model_diagnostics,
        _model_confidence,
        _confidence_available,
        _original_gray,
        detail_geometry_mode,
        _model_views,
    ) = _build_model_payloads(
        record,
        tuple(build_result.model_specs or ()),
        analysis_max_side=None,
        geometry_mode=build_result.options.geometry_mode,
        point_match_radius=float(build_result.options.point_match_radius),
        boundary_radius=int(getattr(build_result.options, "boundary_radius", 1) or 1),
        confidence_uncertainty_delta=float(
            getattr(build_result.options, "confidence_uncertainty_delta", MODEL_CONFIDENCE_UNCERTAIN_DELTA)
        ),
        point_confidence_radius=int(
            getattr(build_result.options, "point_confidence_radius", POINT_CONFIDENCE_NEIGHBOR_RADIUS)
            or POINT_CONFIDENCE_NEIGHBOR_RADIUS
        ),
        polygon_confidence_summary=str(
            getattr(build_result.options, "polygon_confidence_summary", POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)
            or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
        ),
        include_confidence_objects=False,
        include_original_gray=True,
        include_model_confidence=False,
        include_pairwise_metrics=False,
        include_model_diagnostics=False,
        include_structure_details=False,
        include_model_output_probabilities=True,
        include_source_grays=False,
    )
    if len(masks) < 2:
        return None
    return _build_frame_comparison_result(
        frame_id=str(record.key),
        model_specs=tuple(build_result.model_specs or ()),
        probabilities_by_model=output_probabilities or probabilities,
        masks_by_model={str(key): np.asarray(value, dtype=bool) for key, value in masks.items()},
        geometry_mode=detail_geometry_mode,
        threshold=float(getattr(build_result.options, "mask_threshold", 0.5) or 0.5),
        consensus_threshold=0.5,
        connectivity=8,
        compute_level="standard",
    )


def _export_pair_model_ids(
    build_result: BuildResult, maps: dict[str, np.ndarray] | None = None
) -> tuple[str, str] | None:
    ordered = [str(spec.model_id) for spec in tuple(build_result.model_specs or ())]
    if maps is not None:
        ordered = [model_id for model_id in ordered if model_id in maps]
    if len(ordered) < 2:
        return None
    return ordered[0], ordered[1]


def _load_export_payload(record: FrameRecord, build_result: BuildResult) -> dict[str, object]:
    (
        probabilities,
        output_probabilities,
        masks,
        _source_grays,
        _model_structures,
        _model_diagnostics,
        _model_confidence,
        _confidence_available,
        original_gray,
        detail_geometry_mode,
        _model_views,
    ) = _build_model_payloads(
        record,
        tuple(build_result.model_specs or ()),
        analysis_max_side=None,
        geometry_mode=build_result.options.geometry_mode,
        point_match_radius=float(build_result.options.point_match_radius),
        boundary_radius=int(getattr(build_result.options, "boundary_radius", 1) or 1),
        confidence_uncertainty_delta=float(
            getattr(build_result.options, "confidence_uncertainty_delta", MODEL_CONFIDENCE_UNCERTAIN_DELTA)
        ),
        point_confidence_radius=int(
            getattr(build_result.options, "point_confidence_radius", POINT_CONFIDENCE_NEIGHBOR_RADIUS)
            or POINT_CONFIDENCE_NEIGHBOR_RADIUS
        ),
        polygon_confidence_summary=str(
            getattr(build_result.options, "polygon_confidence_summary", POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)
            or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
        ),
        include_confidence_objects=False,
        include_original_gray=True,
        include_model_confidence=False,
        include_pairwise_metrics=False,
        include_model_diagnostics=False,
        include_structure_details=False,
        include_model_output_probabilities=True,
        include_source_grays=False,
    )
    return {
        "probabilities": probabilities,
        "output_probabilities": output_probabilities,
        "masks": masks,
        "original_gray": original_gray,
        "geometry_mode": detail_geometry_mode,
    }


def _probability_for_export(payload: dict[str, object], model_id: str) -> np.ndarray | None:
    output_probabilities = payload.get("output_probabilities") or {}
    probabilities = payload.get("probabilities") or {}
    if isinstance(output_probabilities, dict) and model_id in output_probabilities:
        return np.asarray(output_probabilities[model_id], dtype=np.float32)
    if isinstance(probabilities, dict) and model_id in probabilities:
        return np.asarray(probabilities[model_id], dtype=np.float32)
    return None


def _input_edge_support_export(original_gray: np.ndarray | None) -> np.ndarray | None:
    if original_gray is None:
        return None
    values = np.asarray(original_gray, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        return np.zeros_like(values, dtype=bool)
    low = float(np.percentile(finite, 5.0))
    high = float(np.percentile(finite, 95.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-6:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    grad_x = np.zeros_like(normalized, dtype=np.float32)
    grad_y = np.zeros_like(normalized, dtype=np.float32)
    grad_x[:, :-1] = np.abs(normalized[:, 1:] - normalized[:, :-1])
    grad_y[:-1, :] = np.abs(normalized[1:, :] - normalized[:-1, :])
    gradient = np.hypot(grad_x, grad_y)
    positive = gradient[gradient > 0.0]
    if positive.size > 0:
        threshold = float(np.percentile(positive, 85.0))
        threshold = max(threshold, float(np.mean(positive) + 0.5 * np.std(positive)), 0.04)
    else:
        threshold = 1.0
    support = np.asarray(gradient >= threshold, dtype=bool)
    if not np.any(support):
        support = np.asarray(gradient > 0.0, dtype=bool)
    return np.asarray(_distance_transform(~support) <= 1.0, dtype=bool)


def _input_output_heatmap_export(
    payload: dict[str, object], model_id: str, build_result: BuildResult
) -> np.ndarray | None:
    masks = payload.get("masks") or {}
    original_gray = payload.get("original_gray")
    if not isinstance(masks, dict) or model_id not in masks or original_gray is None:
        return None
    input_edges = _input_edge_support_export(np.asarray(original_gray, dtype=np.uint8))
    output_boundary = np.asarray(_boundary_mask(np.asarray(masks[model_id], dtype=bool)), dtype=bool)
    if input_edges is None or input_edges.shape != output_boundary.shape:
        return None
    if not np.any(input_edges) and not np.any(output_boundary):
        return np.zeros_like(output_boundary, dtype=np.float32)
    tolerance = max(2.0, float(getattr(build_result.options, "boundary_radius", 1) or 1.0) + 1.5)
    input_band = np.asarray(_distance_transform(~input_edges) <= 1.0, dtype=bool)
    output_band = np.asarray(_distance_transform(~output_boundary) <= 1.0, dtype=bool)
    dist_to_input = np.asarray(_distance_transform(~input_band), dtype=np.float32)
    dist_to_output = np.asarray(_distance_transform(~output_band), dtype=np.float32)
    output_mismatch = np.clip(dist_to_input / float(tolerance), 0.0, 1.0) * np.asarray(
        output_boundary, dtype=np.float32
    )
    input_miss = np.clip(dist_to_output / float(tolerance), 0.0, 1.0) * np.asarray(input_edges, dtype=np.float32)
    return np.clip(np.maximum(output_mismatch, input_miss), 0.0, 1.0).astype(np.float32, copy=False)


def _result_kind_export_array(
    build_result: BuildResult,
    payload: dict[str, object],
    result_kind: str,
) -> tuple[np.ndarray, str] | None:
    masks = payload.get("masks") or {}
    if not isinstance(masks, dict):
        return None
    pair = _export_pair_model_ids(build_result, masks)
    title_by_kind = {
        "diff": "Comparison difference",
        "iou": "IoU overlap",
        "dice": "Dice overlap",
        "bce": "BCE heatmap",
        "boundary": "Boundary difference",
        "input_output": "Input-output heatmap",
    }
    if result_kind == "input_output":
        model_id = pair[0] if pair is not None else next(iter(masks.keys()), None)
        if model_id is None:
            return None
        heatmap = _input_output_heatmap_export(payload, str(model_id), build_result)
        return None if heatmap is None else (heatmap, title_by_kind[result_kind])
    if pair is None:
        return None
    first_id, second_id = pair
    first_mask = np.asarray(masks[first_id], dtype=bool)
    second_mask = np.asarray(masks[second_id], dtype=bool)
    if result_kind == "diff":
        heatmap, _score = compute_comparison(first_mask, second_mask, ComparisonMode.DISAGREEMENT)
        return heatmap, title_by_kind[result_kind]
    if result_kind in {"iou", "dice"}:
        return np.logical_and(first_mask, second_mask).astype(np.float32), title_by_kind[result_kind]
    if result_kind == "boundary":
        first_boundary = np.asarray(_boundary_mask(first_mask), dtype=bool)
        second_boundary = np.asarray(_boundary_mask(second_mask), dtype=bool)
        heatmap, _score = compute_comparison(first_boundary, second_boundary, ComparisonMode.DISAGREEMENT)
        return heatmap, title_by_kind[result_kind]
    if result_kind == "bce":
        first_prob = _probability_for_export(payload, first_id)
        second_prob = _probability_for_export(payload, second_id)
        if first_prob is None or second_prob is None:
            return None
        heatmap, _score = compute_comparison(first_prob, second_prob, ComparisonMode.BCE)
        return heatmap, title_by_kind[result_kind]
    return None


def _comparison_layer_export_array(
    build_result: BuildResult,
    payload: dict[str, object],
    layer_id: str,
) -> tuple[np.ndarray, str] | None:
    masks = payload.get("masks") or {}
    if not isinstance(masks, dict):
        return None
    title_by_layer = {
        "mask_common": "A and B",
        "mask_a_only": "A only",
        "mask_b_only": "B only",
        "mask_xor": "A xor B",
        "mask_union": "A or B",
        "boundary_a": "Boundary A",
        "boundary_b": "Boundary B",
        "soft_abs_difference": "|P_A - P_B|",
        "soft_signed_difference": "P_A - P_B",
        "threshold_crossing_map": "Threshold crossing",
        "vote_map": "Vote map",
        "consensus_mask": "Consensus mask",
        "ensemble_uncertainty": "Ensemble uncertainty",
    }
    pair = _export_pair_model_ids(build_result, masks)
    if pair is not None:
        first_id, second_id = pair
        first_mask = np.asarray(masks[first_id], dtype=bool)
        second_mask = np.asarray(masks[second_id], dtype=bool)
        common = first_mask & second_mask
        if layer_id == "mask_common":
            return common.astype(np.float32), title_by_layer[layer_id]
        if layer_id == "mask_a_only":
            return (first_mask & ~second_mask).astype(np.float32), title_by_layer[layer_id]
        if layer_id == "mask_b_only":
            return (second_mask & ~first_mask).astype(np.float32), title_by_layer[layer_id]
        if layer_id == "mask_xor":
            return np.logical_xor(first_mask, second_mask).astype(np.float32), title_by_layer[layer_id]
        if layer_id == "mask_union":
            return np.logical_or(first_mask, second_mask).astype(np.float32), title_by_layer[layer_id]
        if layer_id == "boundary_a":
            return np.asarray(_boundary_mask(first_mask), dtype=np.float32), title_by_layer[layer_id]
        if layer_id == "boundary_b":
            return np.asarray(_boundary_mask(second_mask), dtype=np.float32), title_by_layer[layer_id]
        if layer_id in {"soft_abs_difference", "soft_signed_difference", "threshold_crossing_map"}:
            first_prob = _probability_for_export(payload, first_id)
            second_prob = _probability_for_export(payload, second_id)
            if first_prob is None or second_prob is None:
                return None
            if layer_id == "soft_abs_difference":
                return np.abs(first_prob - second_prob).astype(np.float32), title_by_layer[layer_id]
            if layer_id == "soft_signed_difference":
                return np.asarray((first_prob - second_prob + 1.0) * 0.5, dtype=np.float32), title_by_layer[layer_id]
            threshold = float(getattr(build_result.options, "mask_threshold", 0.5) or 0.5)
            return np.logical_xor(first_prob >= threshold, second_prob >= threshold).astype(np.float32), title_by_layer[
                layer_id
            ]
    if len(masks) >= 2 and layer_id in {"vote_map", "consensus_mask", "ensemble_uncertainty"}:
        ordered_ids = [
            str(spec.model_id) for spec in tuple(build_result.model_specs or ()) if str(spec.model_id) in masks
        ]
        if not ordered_ids:
            ordered_ids = [str(key) for key in masks.keys()]
        stack = np.stack([np.asarray(masks[model_id], dtype=np.float32) for model_id in ordered_ids], axis=0)
        vote_map = np.mean(stack, axis=0, dtype=np.float32)
        if layer_id == "vote_map":
            return vote_map, title_by_layer[layer_id]
        consensus = vote_map >= 0.5
        if layer_id == "consensus_mask":
            return consensus.astype(np.float32), title_by_layer[layer_id]
        uncertainty = np.asarray(1.0 - np.abs(vote_map - 0.5) * 2.0, dtype=np.float32)
        return uncertainty, title_by_layer[layer_id]
    return None


def _export_pair_specs(build_result: BuildResult, record: FrameRecord) -> tuple[ModelSpec, ModelSpec] | None:
    pairs = [
        spec
        for spec in tuple(build_result.model_specs or ())
        if bool((record.model_mask_paths or {}).get(str(spec.model_id)))
    ]
    if len(pairs) < 2:
        return None
    return pairs[0], pairs[1]


def _load_export_mask_for_spec(
    record: FrameRecord, spec: ModelSpec, target_shape: tuple[int, int] | None = None
) -> np.ndarray | None:
    path_text = str((record.model_mask_paths or {}).get(str(spec.model_id)) or "")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    gray = _load_export_grayscale_image(path, target_shape=target_shape)
    return np.asarray(_mask_from_gray(gray, threshold=float(getattr(spec, "threshold", 0.5) or 0.5)), dtype=bool)


def _load_export_probability_for_spec(
    record: FrameRecord, spec: ModelSpec, target_shape: tuple[int, int] | None = None
) -> np.ndarray | None:
    model_id = str(spec.model_id)
    path_text = str(
        (record.model_prob_paths or {}).get(model_id) or (record.model_mask_paths or {}).get(model_id) or ""
    )
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    gray = _load_export_grayscale_image(path, target_shape=target_shape)
    return np.asarray(_prob_from_gray(gray), dtype=np.float32)


def _load_export_mask_pair(
    record: FrameRecord, build_result: BuildResult
) -> tuple[str, str, np.ndarray, np.ndarray] | None:
    pair = _export_pair_specs(build_result, record)
    if pair is None:
        return None
    first_spec, second_spec = pair
    first = _load_export_mask_for_spec(record, first_spec)
    if first is None:
        return None
    second = _load_export_mask_for_spec(record, second_spec, target_shape=tuple(int(v) for v in first.shape))
    if second is None:
        return None
    return str(first_spec.model_id), str(second_spec.model_id), first, second


def _load_export_probability_pair(
    record: FrameRecord, build_result: BuildResult
) -> tuple[str, str, np.ndarray, np.ndarray] | None:
    pair = _export_pair_specs(build_result, record)
    if pair is None:
        return None
    first_spec, second_spec = pair
    first = _load_export_probability_for_spec(record, first_spec)
    if first is None:
        return None
    second = _load_export_probability_for_spec(record, second_spec, target_shape=tuple(int(v) for v in first.shape))
    if second is None:
        return None
    return str(first_spec.model_id), str(second_spec.model_id), first, second


def _export_confidence_pair_specs(build_result: BuildResult, record: FrameRecord) -> tuple[ModelSpec, ModelSpec] | None:
    pairs = [
        spec
        for spec in tuple(build_result.model_specs or ())
        if bool((record.model_prob_paths or {}).get(str(spec.model_id)))
    ]
    if len(pairs) < 2:
        return None
    return pairs[0], pairs[1]


def _load_export_confidence_for_spec(
    record: FrameRecord, spec: ModelSpec, target_shape: tuple[int, int] | None = None
) -> np.ndarray | None:
    path_text = str((record.model_prob_paths or {}).get(str(spec.model_id)) or "")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    gray = _load_export_grayscale_image(path, target_shape=target_shape)
    return np.asarray(_prob_from_gray(gray), dtype=np.float32)


def _load_export_confidence_pair(
    record: FrameRecord, build_result: BuildResult
) -> tuple[str, str, np.ndarray, np.ndarray] | None:
    pair = _export_confidence_pair_specs(build_result, record)
    if pair is None:
        return None
    first_spec, second_spec = pair
    first = _load_export_confidence_for_spec(record, first_spec)
    if first is None:
        return None
    second = _load_export_confidence_for_spec(record, second_spec, target_shape=tuple(int(v) for v in first.shape))
    if second is None:
        return None
    return str(first_spec.model_id), str(second_spec.model_id), first, second


def _load_export_result_confidence_for_model(
    record: FrameRecord, build_result: BuildResult, model_id: str
) -> tuple[np.ndarray, np.ndarray] | None:
    spec = next(
        (candidate for candidate in tuple(build_result.model_specs or ()) if str(candidate.model_id) == str(model_id)),
        None,
    )
    if spec is None:
        return None
    mask = _load_export_mask_for_spec(record, spec)
    if mask is None:
        return None
    confidence = _load_export_confidence_for_spec(record, spec, target_shape=tuple(int(v) for v in mask.shape))
    if confidence is None:
        return None
    return np.asarray(mask, dtype=bool), np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)


def _result_confidence_diagnostic_map(mask: np.ndarray, confidence: np.ndarray, kind: str) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    confidence_unit = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    if confidence_unit.shape != mask_bool.shape:
        return np.zeros_like(mask_bool, dtype=np.float32)
    uncertainty = build_model_uncertainty(confidence_unit)
    low_confidence = 1.0 - confidence_unit
    distance_inside = np.asarray(_distance_transform(mask_bool), dtype=np.float32)
    interior_weight = np.clip((distance_inside - 1.0) / 2.5, 0.0, 1.0)
    interior_weight *= np.asarray(mask_bool, dtype=np.float32)

    def focused(raw: np.ndarray, support: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(raw, dtype=np.float32) * np.asarray(support, dtype=np.float32), 0.0, 1.0)
        return np.clip((values - 0.06) / 0.94, 0.0, 1.0)

    if kind == "bad_inside":
        return focused(np.maximum(uncertainty, low_confidence), interior_weight)
    if kind == "conflict":
        return focused(low_confidence, interior_weight)
    boundary = np.asarray(_boundary_mask(mask_bool), dtype=bool)
    if kind == "boundary_uncertainty":
        band = np.asarray(_distance_transform(~boundary) <= 2.0, dtype=bool) if np.any(boundary) else boundary
        return np.clip(np.asarray(band, dtype=np.float32) * uncertainty, 0.0, 1.0)
    if kind == "transition_uncertainty":
        band = np.asarray(_distance_transform(~boundary) <= 5.0, dtype=bool) if np.any(boundary) else boundary
        return np.clip(np.asarray(band, dtype=np.float32) * uncertainty, 0.0, 1.0)
    return np.zeros_like(mask_bool, dtype=np.float32)


def _fast_result_confidence_export_array(
    record: FrameRecord, build_result: BuildResult, kind: str, model_id: str
) -> tuple[np.ndarray, str] | None:
    pair = _load_export_result_confidence_for_model(record, build_result, model_id)
    if pair is None:
        return None
    mask, confidence = pair
    title_by_kind = {
        "bad_inside": "Suspicious interior by confidence",
        "boundary_uncertainty": "Uncertain result boundary",
        "conflict": "Result-confidence conflict",
        "transition_uncertainty": "Uncertain transition around result",
    }
    title = title_by_kind.get(str(kind), str(kind))
    return _result_confidence_diagnostic_map(mask, confidence, kind), title


def _fast_grid_cell_defect_export_array(
    record: FrameRecord, build_result: BuildResult, model_id: str
) -> tuple[np.ndarray, str] | None:
    spec = next(
        (candidate for candidate in tuple(build_result.model_specs or ()) if str(candidate.model_id) == str(model_id)),
        None,
    )
    if spec is None:
        return None
    path_text = str((record.model_mask_paths or {}).get(str(model_id)) or "")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    gray = _load_export_grayscale_image(path)
    result = detect_grid_cell_anomalies(gray, frame_id=str(record.key), frame_path=str(path))
    height = max(1, int(getattr(result, "image_height", gray.shape[0] if gray.ndim == 2 else 1) or 1))
    width = max(1, int(getattr(result, "image_width", gray.shape[1] if gray.ndim == 2 else 1) or 1))
    intensity = np.zeros((height, width), dtype=np.float32)
    for cell in getattr(result, "per_cell_results", getattr(result, "cells", ())) or ():
        if str(getattr(cell, "status", "")) == "normal":
            continue
        x = max(0, min(width - 1, int(getattr(cell, "left", 0))))
        y = max(0, min(height - 1, int(getattr(cell, "top", 0))))
        w = max(1, int(getattr(cell, "width", 1)))
        h = max(1, int(getattr(cell, "height", 1)))
        x1 = min(width, x + w)
        y1 = min(height, y + h)
        if x1 > x and y1 > y:
            intensity[y:y1, x:x1] = np.maximum(intensity[y:y1, x:x1], float(getattr(cell, "score", 0.0)))
    title = f"{str(getattr(spec, 'display_name', '') or model_id)} grid defects"
    return intensity, title


def grid_cell_defect_check_mask(
    result: GridFrameAnalysisResult,
    *,
    enabled_reason_types: tuple[str, ...] | list[str] | set[str] | None = None,
) -> np.ndarray:
    """Render detected grid-cell errors as a black/white export mask."""

    height = max(1, int(getattr(result, "image_height", 0) or 1))
    width = max(1, int(getattr(result, "image_width", 0) or 1))
    mask = np.zeros((height, width), dtype=np.uint8)
    enabled_set = (
        None if enabled_reason_types is None else {str(reason) for reason in enabled_reason_types if str(reason)}
    )
    for cell in getattr(result, "per_cell_results", getattr(result, "cells", ())) or ():
        if str(getattr(cell, "status", "") or "") == "normal":
            continue
        reasons = tuple(str(reason) for reason in (getattr(cell, "reasons", ()) or ()) if str(reason))
        if enabled_set is not None and reasons and not any(reason in enabled_set for reason in reasons):
            continue
        x = max(0, min(width - 1, int(getattr(cell, "left", 0))))
        y = max(0, min(height - 1, int(getattr(cell, "top", 0))))
        w = max(1, int(getattr(cell, "width", 1)))
        h = max(1, int(getattr(cell, "height", 1)))
        x1 = min(width, x + w)
        y1 = min(height, y + h)
        if x1 > x and y1 > y:
            mask[y:y1, x:x1] = 255
    return mask


def grid_cell_presence_mask(result: GridFrameAnalysisResult) -> np.ndarray:
    """Render every detected grid cell in white on a black background."""

    height = max(1, int(getattr(result, "image_height", 0) or 1))
    width = max(1, int(getattr(result, "image_width", 0) or 1))
    mask = np.zeros((height, width), dtype=np.uint8)
    for cell in getattr(result, "per_cell_results", getattr(result, "cells", ())) or ():
        x = max(0, min(width - 1, int(getattr(cell, "left", 0))))
        y = max(0, min(height - 1, int(getattr(cell, "top", 0))))
        w = max(1, int(getattr(cell, "width", 1)))
        h = max(1, int(getattr(cell, "height", 1)))
        x1 = min(width, x + w)
        y1 = min(height, y + h)
        if x1 > x and y1 > y:
            mask[y:y1, x:x1] = 255
    return mask


def _grid_check_record_source_path(record: FrameRecord) -> Path | None:
    model_masks = getattr(record, "model_mask_paths", {}) or {}
    path_text = str(next(iter(model_masks.values())) or "") if model_masks else ""
    if not path_text:
        path_text = str(
            getattr(record, "first_path", "")
            or getattr(record, "base_path", "")
            or getattr(record, "original_path", "")
            or ""
        )
    return Path(path_text) if path_text else None


def _normalize_grid_check_export_format(image_format: str | None) -> tuple[str, str]:
    value = str(image_format or "bmp").strip().lower().lstrip(".")
    if value in {"jpg", "jpeg"}:
        return "JPG", "jpg"
    if value == "png":
        return "PNG", "png"
    if value == "bmp":
        return "BMP", "bmp"
    raise ValueError(f"Unsupported grid defect export format: {image_format}")


def _grid_check_export_folder_name(records: tuple[FrameRecord, ...], extension: str) -> str:
    first_record = next((record for record in records if record is not None), None)
    source_path = _grid_check_record_source_path(first_record) if first_record is not None else None
    if source_path is not None and source_path.name:
        source_stem = source_path.stem
    elif first_record is not None:
        source_stem = Path(
            str(getattr(first_record, "display_name", "") or getattr(first_record, "key", "") or "frames")
        ).stem
    else:
        source_stem = "frames"
    stem_without_digits = "".join(char for char in str(source_stem) if not char.isdigit())
    clean_stem = _safe_export_name(stem_without_digits, fallback="frames")
    return _safe_export_name(f"check_{clean_stem}_{extension}", fallback=f"check_frames_{extension}")


def export_grid_cell_defect_bmps(
    build_result: BuildResult,
    results_by_key: dict[str, GridFrameAnalysisResult],
    destination: Path | str,
    *,
    records: tuple[FrameRecord, ...] | list[FrameRecord] | None = None,
    render_record_keys: tuple[str, ...] | list[str] | set[str] | None = None,
    image_format: str = "bmp",
    folder_name: str | None = None,
    enabled_reason_types: tuple[str, ...] | list[str] | set[str] | None = None,
    progress_callback=None,
    cancel_check=None,
) -> dict[str, object]:
    """Export computed grid-cell error highlights as black/white masks into the export folder."""

    payloads = {str(key): value for key, value in (results_by_key or {}).items()}
    qt_format, extension = _normalize_grid_check_export_format(image_format)
    source_records = tuple(records if records is not None else (getattr(build_result, "records", ()) or ()))
    render_key_set = None if render_record_keys is None else {str(key) for key in render_record_keys if str(key)}
    missing_records = tuple(
        record for record in source_records if str(getattr(record, "key", "") or "") not in payloads
    )
    export_records = tuple(record for record in source_records if str(getattr(record, "key", "") or "") in payloads)
    if not export_records:
        raise ValueError("No computed grid defect results are available for export.")

    target_folder_name = str(folder_name or _grid_check_export_folder_name(export_records, extension))
    destination_path = Path(destination) / target_folder_name
    destination_path.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, str]] = []
    errors: list[str] = [
        f"{str(getattr(record, 'display_name', '') or getattr(record, 'key', '') or 'frame')}: result is unavailable"
        for record in missing_records
    ]
    used_names: set[str] = set()
    total = len(export_records)
    if progress_callback is not None:
        progress_callback(0, total, "")

    for index, record in enumerate(export_records, start=1):
        frame_name = str(getattr(record, "display_name", "") or getattr(record, "key", "") or f"frame_{index:06d}")
        if cancel_check is not None and cancel_check():
            errors.append("Export cancelled")
            break
        try:
            result = payloads.get(str(getattr(record, "key", "") or ""))
            if result is None:
                errors.append(f"{frame_name}: result is unavailable")
                continue
            source_path = _grid_check_record_source_path(record)
            source_stem = source_path.stem if source_path is not None and source_path.name else Path(frame_name).stem
            base_name = _safe_export_name(source_stem, fallback=f"frame_{index:06d}")
            if base_name in used_names:
                base_name = _unique_export_folder_name(base_name, used_names)
            else:
                used_names.add(base_name)
            target_path = destination_path / f"{base_name}.{extension}"
            record_key = str(getattr(record, "key", "") or "")
            if render_key_set is not None and record_key not in render_key_set:
                height = max(1, int(getattr(result, "image_height", 0) or 1))
                width = max(1, int(getattr(result, "image_width", 0) or 1))
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = grid_cell_defect_check_mask(result, enabled_reason_types=enabled_reason_types)
            quality = 100 if qt_format == "JPG" else -1
            if not _grayscale_array_to_qimage(mask).save(str(target_path), qt_format, quality):
                errors.append(f"{frame_name}: failed to save {qt_format}")
                continue
            exported.append(
                {
                    "record_key": str(getattr(record, "key", "") or ""),
                    "record_name": frame_name,
                    "source": str(source_path) if source_path is not None else "",
                    "destination": str(target_path),
                }
            )
        except Exception as error:
            errors.append(f"{frame_name}: {error}")
        finally:
            if progress_callback is not None:
                progress_callback(index, total, frame_name)

    return {
        "exported_count": len(exported),
        "skipped_count": len(errors),
        "destination": str(destination_path),
        "format": qt_format,
        "extension": extension,
        "errors": tuple(errors),
        "files": tuple(exported),
    }


def export_grid_cell_defect_canvas(
    build_result: BuildResult,
    results_by_key: dict[str, GridFrameAnalysisResult],
    destination: Path | str,
    *,
    canvas_width: int,
    canvas_height: int,
    frames_per_row: int,
    records: tuple[FrameRecord, ...] | list[FrameRecord] | None = None,
    render_record_keys: tuple[str, ...] | list[str] | set[str] | None = None,
    enabled_reason_types: tuple[str, ...] | list[str] | set[str] | None = None,
    preserve_aspect_ratio: bool = True,
    overlay_errors_on_source_mask: bool = False,
    file_name: str = "check_matrix.bmp",
    progress_callback=None,
    cancel_check=None,
) -> dict[str, object]:
    """Combine grid-cell error masks into one exact-size binary BMP canvas."""

    width = int(canvas_width)
    height = int(canvas_height)
    columns = int(frames_per_row)
    if width <= 0 or height <= 0:
        raise ValueError("Canvas width and height must be positive.")
    if columns <= 0:
        raise ValueError("Frames per row must be positive.")
    bmp_row_stride = ((width + 3) // 4) * 4
    if bmp_row_stride * height + 1078 > 0xFFFFFFFF:
        raise ValueError("The requested canvas is too large for the BMP format.")

    source_records = tuple(records if records is not None else (getattr(build_result, "records", ()) or ()))
    if not source_records:
        raise ValueError("No frame records are available for canvas export.")
    payloads = {str(key): value for key, value in (results_by_key or {}).items()}
    if not payloads:
        raise ValueError("No computed grid defect results are available for export.")

    render_key_set = None if render_record_keys is None else {str(key) for key in render_record_keys if str(key)}
    rows = max(1, math.ceil(len(source_records) / columns))
    canvas_shape = (height, width, 3) if overlay_errors_on_source_mask else (height, width)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    errors: list[str] = []
    rendered_count = 0
    total = len(source_records)
    if progress_callback is not None:
        progress_callback(0, total, "")

    for index, record in enumerate(source_records):
        frame_name = str(getattr(record, "display_name", "") or getattr(record, "key", "") or f"frame_{index + 1:06d}")
        if cancel_check is not None and cancel_check():
            return {
                "exported_count": 0,
                "rendered_count": rendered_count,
                "skipped_count": len(errors),
                "destination": "",
                "format": "BMP",
                "extension": "bmp",
                "canvas_size": (width, height),
                "errors": tuple(errors),
                "cancelled": True,
            }
        try:
            record_key = str(getattr(record, "key", "") or "")
            result = payloads.get(record_key)
            if result is None:
                errors.append(f"{frame_name}: result is unavailable")
                continue

            row = index // columns
            column = index % columns
            x0 = column * width // columns
            x1 = (column + 1) * width // columns
            y0 = row * height // rows
            y1 = (row + 1) * height // rows
            slot_width = x1 - x0
            slot_height = y1 - y0
            if slot_width <= 0 or slot_height <= 0:
                errors.append(f"{frame_name}: canvas slot has zero size")
                continue
            if render_key_set is not None and record_key not in render_key_set:
                continue

            mask = grid_cell_defect_check_mask(result, enabled_reason_types=enabled_reason_types)
            source_height, source_width = mask.shape
            if preserve_aspect_ratio and not overlay_errors_on_source_mask:
                scale = min(slot_width / source_width, slot_height / source_height)
                target_width = max(1, min(slot_width, int(round(source_width * scale))))
                target_height = max(1, min(slot_height, int(round(source_height * scale))))
            else:
                target_width = slot_width
                target_height = slot_height
            resized = resize_grayscale_image(mask, (target_height, target_width))
            binary = np.where(resized > 0, 255, 0).astype(np.uint8)
            target_x = x0 + (slot_width - target_width) // 2
            target_y = y0 + (slot_height - target_height) // 2
            target = canvas[target_y : target_y + target_height, target_x : target_x + target_width]
            if overlay_errors_on_source_mask:
                source_mask = grid_cell_presence_mask(result)
                source_resized = resize_grayscale_image(source_mask, (target_height, target_width))
                source_binary = np.where(source_resized > 0, 255, 0).astype(np.uint8)
                tile = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                tile[source_binary > 0] = (255, 255, 255)
                tile[binary > 0] = (255, 0, 0)
                target[:] = tile
            else:
                np.maximum(target, binary, out=target)
            rendered_count += 1
        except Exception as error:
            errors.append(f"{frame_name}: {error}")
        finally:
            if progress_callback is not None:
                progress_callback(index + 1, total, frame_name)

    normalized_name = Path(str(file_name or "check_matrix.bmp")).name
    if Path(normalized_name).suffix.lower() != ".bmp":
        normalized_name = f"{Path(normalized_name).stem}.bmp"
    destination_path = Path(destination) / normalized_name
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image = _rgb_array_to_qimage(canvas) if overlay_errors_on_source_mask else _grayscale_array_to_qimage(canvas)
    if not image.save(str(destination_path), "BMP"):
        raise OSError(f"Failed to save BMP canvas: {destination_path}")
    return {
        "exported_count": 1,
        "rendered_count": rendered_count,
        "skipped_count": len(errors),
        "destination": str(destination_path),
        "format": "BMP",
        "extension": "bmp",
        "canvas_size": (width, height),
        "errors": tuple(errors),
        "cancelled": False,
    }


def _fast_result_kind_export_array(
    record: FrameRecord, build_result: BuildResult, result_kind: str
) -> tuple[np.ndarray, str] | None:
    title_by_kind = {
        "diff": "Comparison difference",
        "iou": "IoU overlap",
        "dice": "Dice overlap",
        "bce": "BCE heatmap",
        "boundary": "Boundary difference",
        "input_output": "Input-output heatmap",
        "confidence_difference": "Confidence difference",
        "confidence_bce": "Confidence BCE heatmap",
        "confidence_threshold_crossing": "Confidence threshold crossing",
    }
    if result_kind in {"confidence_difference", "confidence_bce", "confidence_threshold_crossing"}:
        confidence_pair = _load_export_confidence_pair(record, build_result)
        if confidence_pair is None:
            return None
        _first_id, _second_id, first_confidence, second_confidence = confidence_pair
        if result_kind == "confidence_difference":
            return np.abs(first_confidence - second_confidence).astype(np.float32), title_by_kind[result_kind]
        if result_kind == "confidence_bce":
            heatmap, _score = compute_comparison(first_confidence, second_confidence, ComparisonMode.BCE)
            return heatmap, title_by_kind[result_kind]
        threshold = float(getattr(build_result.options, "mask_threshold", 0.5) or 0.5)
        return np.logical_xor(first_confidence >= threshold, second_confidence >= threshold).astype(
            np.float32
        ), title_by_kind[result_kind]
    if result_kind == "bce":
        probability_pair = _load_export_probability_pair(record, build_result)
        if probability_pair is None:
            return None
        _first_id, _second_id, first_prob, second_prob = probability_pair
        heatmap, _score = compute_comparison(first_prob, second_prob, ComparisonMode.BCE)
        return heatmap, title_by_kind[result_kind]
    mask_pair = _load_export_mask_pair(record, build_result)
    if mask_pair is None:
        return None
    first_id, _second_id, first_mask, second_mask = mask_pair
    if result_kind == "diff":
        return np.logical_xor(first_mask, second_mask).astype(np.float32), title_by_kind[result_kind]
    if result_kind in {"iou", "dice"}:
        return np.logical_and(first_mask, second_mask).astype(np.float32), title_by_kind[result_kind]
    if result_kind == "boundary":
        return np.logical_xor(_boundary_mask(first_mask), _boundary_mask(second_mask)).astype(
            np.float32
        ), title_by_kind[result_kind]
    if result_kind == "input_output":
        original_path = str(getattr(record, "original_path", "") or getattr(record, "base_path", "") or "")
        if not original_path or not Path(original_path).is_file():
            return None
        original_gray = _load_export_grayscale_image(
            original_path, target_shape=tuple(int(v) for v in first_mask.shape)
        )
        payload = {"masks": {first_id: first_mask}, "original_gray": original_gray}
        heatmap = _input_output_heatmap_export(payload, str(first_id), build_result)
        return None if heatmap is None else (heatmap, title_by_kind[result_kind])
    return None


def _fast_comparison_layer_export_array(
    record: FrameRecord, build_result: BuildResult, layer_id: str
) -> tuple[np.ndarray, str] | None:
    title_by_layer = {
        "mask_common": "A and B",
        "mask_a_only": "A only",
        "mask_b_only": "B only",
        "mask_xor": "A xor B",
        "mask_union": "A or B",
        "boundary_a": "Boundary A",
        "boundary_b": "Boundary B",
        "soft_abs_difference": "|P_A - P_B|",
        "soft_signed_difference": "P_A - P_B",
        "threshold_crossing_map": "Threshold crossing",
        "vote_map": "Vote map",
        "consensus_mask": "Consensus mask",
        "ensemble_uncertainty": "Ensemble uncertainty",
    }
    if layer_id in {"soft_abs_difference", "soft_signed_difference", "threshold_crossing_map"}:
        probability_pair = _load_export_probability_pair(record, build_result)
        if probability_pair is None:
            return None
        _first_id, _second_id, first_prob, second_prob = probability_pair
        if layer_id == "soft_abs_difference":
            return np.abs(first_prob - second_prob).astype(np.float32), title_by_layer[layer_id]
        if layer_id == "soft_signed_difference":
            return np.asarray((first_prob - second_prob + 1.0) * 0.5, dtype=np.float32), title_by_layer[layer_id]
        threshold = float(getattr(build_result.options, "mask_threshold", 0.5) or 0.5)
        return np.logical_xor(first_prob >= threshold, second_prob >= threshold).astype(np.float32), title_by_layer[
            layer_id
        ]

    if layer_id in {"vote_map", "consensus_mask", "ensemble_uncertainty"}:
        masks: list[np.ndarray] = []
        target_shape: tuple[int, int] | None = None
        for spec in tuple(build_result.model_specs or ()):
            mask = _load_export_mask_for_spec(record, spec, target_shape=target_shape)
            if mask is None:
                continue
            if target_shape is None:
                target_shape = tuple(int(v) for v in mask.shape)
            masks.append(mask.astype(np.float32))
        if len(masks) < 2:
            return None
        vote_map = np.mean(np.stack(masks, axis=0), axis=0, dtype=np.float32)
        if layer_id == "vote_map":
            return vote_map, title_by_layer[layer_id]
        if layer_id == "consensus_mask":
            return (vote_map >= 0.5).astype(np.float32), title_by_layer[layer_id]
        return np.asarray(1.0 - np.abs(vote_map - 0.5) * 2.0, dtype=np.float32), title_by_layer[layer_id]

    mask_pair = _load_export_mask_pair(record, build_result)
    if mask_pair is None:
        return None
    _first_id, _second_id, first_mask, second_mask = mask_pair
    if layer_id == "mask_common":
        return (first_mask & second_mask).astype(np.float32), title_by_layer[layer_id]
    if layer_id == "mask_a_only":
        return (first_mask & ~second_mask).astype(np.float32), title_by_layer[layer_id]
    if layer_id == "mask_b_only":
        return (second_mask & ~first_mask).astype(np.float32), title_by_layer[layer_id]
    if layer_id == "mask_xor":
        return np.logical_xor(first_mask, second_mask).astype(np.float32), title_by_layer[layer_id]
    if layer_id == "mask_union":
        return np.logical_or(first_mask, second_mask).astype(np.float32), title_by_layer[layer_id]
    if layer_id == "boundary_a":
        return np.asarray(_boundary_mask(first_mask), dtype=np.float32), title_by_layer[layer_id]
    if layer_id == "boundary_b":
        return np.asarray(_boundary_mask(second_mask), dtype=np.float32), title_by_layer[layer_id]
    return None


def available_result_layer_exports(build_result: BuildResult) -> tuple[dict[str, str], ...]:
    """Return exportable result-layer choices for the current build result."""

    choices: list[dict[str, str]] = []
    display_names = _model_display_name_map(build_result)
    records = tuple(build_result.records or ())
    if len(tuple(build_result.model_specs or ())) >= 2:
        choices.extend(
            (
                {
                    "key": "result_kind::diff",
                    "title": "Comparison difference",
                    "title_key": "details.comparison_difference",
                    "group": "detail",
                },
                {
                    "key": "result_kind::iou",
                    "title": "IoU overlap",
                    "title_key": "details.iou_overlap",
                    "group": "detail",
                },
                {
                    "key": "result_kind::dice",
                    "title": "Dice overlap",
                    "title_key": "details.dice_overlap",
                    "group": "detail",
                },
                {
                    "key": "result_kind::bce",
                    "title": "BCE heatmap",
                    "title_key": "details.bce_heatmap",
                    "group": "detail",
                },
                {
                    "key": "result_kind::boundary",
                    "title": "Boundary difference",
                    "title_key": "details.boundary_difference",
                    "group": "detail",
                },
            )
        )
    if any(bool(getattr(record, "original_path", None)) for record in records):
        choices.append(
            {
                "key": "result_kind::input_output",
                "title": "Input-output heatmap",
                "title_key": "details.input_output_heatmap",
                "group": "detail",
            }
        )
    confidence_model_ids = [
        str(spec.model_id)
        for spec in tuple(build_result.model_specs or ())
        if any(bool((record.model_prob_paths or {}).get(str(spec.model_id))) for record in records)
    ]
    if len(confidence_model_ids) >= 2:
        choices.extend(
            (
                {
                    "key": "result_kind::confidence_difference",
                    "title": "Confidence difference",
                    "title_key": "details.confidence_difference",
                    "group": "confidence",
                },
                {
                    "key": "result_kind::confidence_bce",
                    "title": "Confidence BCE heatmap",
                    "title_key": "details.confidence_bce_heatmap",
                    "group": "confidence",
                },
                {
                    "key": "result_kind::confidence_threshold_crossing",
                    "title": "Confidence threshold crossing",
                    "title_key": "details.confidence_threshold_crossing",
                    "group": "confidence",
                },
            )
        )
    for spec in tuple(build_result.model_specs or ()):
        model_id = str(spec.model_id)
        title = str(display_names.get(model_id) or model_id)
        if any(bool((record.model_mask_paths or {}).get(model_id)) for record in records):
            choices.append(
                {
                    "key": f"model_mask::{model_id}",
                    "title": f"{title} mask",
                    "group": "model",
                }
            )
            choices.append(
                {
                    "key": f"grid_cell_defects::{model_id}",
                    "title": f"{title} grid defects",
                    "title_key": "details.grid_cell_defects",
                    "group": "geometry",
                }
            )
        if any(bool((record.model_prob_paths or {}).get(model_id)) for record in records):
            choices.append(
                {
                    "key": f"model_probability::{model_id}",
                    "title": f"{title} confidence",
                    "group": "model_confidence",
                }
            )
        if any(
            bool((record.model_mask_paths or {}).get(model_id)) and bool((record.model_prob_paths or {}).get(model_id))
            for record in records
        ):
            choices.extend(
                (
                    {
                        "key": f"result_confidence::bad_inside::{model_id}",
                        "title": f"{title} suspicious interior by confidence",
                        "title_key": "details.result_confidence_bad_inside",
                        "group": "result_confidence",
                    },
                    {
                        "key": f"result_confidence::conflict::{model_id}",
                        "title": f"{title} result-confidence conflict",
                        "title_key": "details.result_confidence_conflict",
                        "group": "result_confidence",
                    },
                )
            )

    if len(tuple(build_result.model_specs or ())) >= 2:
        choices.extend(
            (
                {"key": "comparison::mask_common", "title": "A and B", "group": "pixel"},
                {"key": "comparison::mask_a_only", "title": "A only", "group": "pixel"},
                {"key": "comparison::mask_b_only", "title": "B only", "group": "pixel"},
                {"key": "comparison::mask_xor", "title": "A xor B", "group": "pixel"},
                {"key": "comparison::mask_union", "title": "A or B", "group": "pixel"},
                {"key": "comparison::boundary_a", "title": "Boundary A", "group": "geometry"},
                {"key": "comparison::boundary_b", "title": "Boundary B", "group": "geometry"},
                {"key": "comparison::soft_abs_difference", "title": "|P_A - P_B|", "group": "soft_confidence"},
                {"key": "comparison::soft_signed_difference", "title": "P_A - P_B", "group": "soft_confidence"},
                {
                    "key": "comparison::threshold_crossing_map",
                    "title": "Threshold crossing",
                    "group": "soft_confidence",
                },
            )
        )
    if len(tuple(build_result.model_specs or ())) > 2:
        choices.extend(
            (
                {"key": "comparison::vote_map", "title": "Vote map", "group": "ensemble"},
                {"key": "comparison::consensus_mask", "title": "Consensus mask", "group": "ensemble"},
                {"key": "comparison::ensemble_uncertainty", "title": "Ensemble uncertainty", "group": "ensemble"},
            )
        )
    return tuple(choices)


def _layer_export_array_for_record(
    build_result: BuildResult,
    record: FrameRecord,
    layer_key: str,
) -> tuple[np.ndarray, str] | None:
    key = str(layer_key or "")
    display_names = _model_display_name_map(build_result)
    if key.startswith("model_mask::"):
        model_id = key.split("::", 1)[1]
        path_text = str((record.model_mask_paths or {}).get(model_id) or "")
        if not path_text:
            return None
        path = Path(path_text)
        if not path.is_file():
            return None
        return _load_export_grayscale_image(path), f"{display_names.get(model_id, model_id)} mask"
    if key.startswith("model_probability::"):
        model_id = key.split("::", 1)[1]
        path_text = str((record.model_prob_paths or {}).get(model_id) or "")
        if not path_text:
            return None
        path = Path(path_text)
        if not path.is_file():
            return None
        return _load_export_grayscale_image(path), f"{display_names.get(model_id, model_id)} confidence"
    if key.startswith("grid_cell_defects::"):
        model_id = key.split("::", 1)[1]
        return _fast_grid_cell_defect_export_array(record, build_result, model_id)
    if key.startswith("result_kind::"):
        result_kind = key.split("::", 1)[1]
        fast = _fast_result_kind_export_array(record, build_result, result_kind)
        if fast is not None:
            return fast
        payload = _load_export_payload(record, build_result)
        return _result_kind_export_array(build_result, payload, result_kind)
    if key.startswith("result_confidence::"):
        parts = key.split("::", 2)
        if len(parts) == 3:
            return _fast_result_confidence_export_array(record, build_result, parts[1], parts[2])
    if key.startswith("comparison::"):
        layer_id = key.split("::", 1)[1]
        fast = _fast_comparison_layer_export_array(record, build_result, layer_id)
        if fast is not None:
            return fast
        payload = _load_export_payload(record, build_result)
        direct = _comparison_layer_export_array(build_result, payload, layer_id)
        if direct is not None:
            return direct
    return None


def _logical_export_layer_folder_stem(layer_key: str, layer_title: str | None = None) -> str:
    key = str(layer_key or "")
    known = {
        "result_kind::diff": "comparison_difference",
        "result_kind::iou": "iou_overlap",
        "result_kind::dice": "dice_overlap",
        "result_kind::bce": "bce_heatmap",
        "result_kind::boundary": "boundary_difference",
        "result_kind::input_output": "input_output_heatmap",
        "result_kind::confidence_difference": "confidence_difference",
        "result_kind::confidence_bce": "confidence_bce_heatmap",
        "result_kind::confidence_threshold_crossing": "confidence_threshold_crossing",
        "comparison::mask_common": "a_and_b",
        "comparison::mask_a_only": "a_only",
        "comparison::mask_b_only": "b_only",
        "comparison::mask_xor": "a_xor_b",
        "comparison::mask_union": "a_or_b",
        "comparison::boundary_a": "boundary_a",
        "comparison::boundary_b": "boundary_b",
        "comparison::soft_abs_difference": "soft_abs_difference",
        "comparison::soft_signed_difference": "soft_signed_difference",
        "comparison::threshold_crossing_map": "threshold_crossing_map",
        "comparison::vote_map": "vote_map",
        "comparison::consensus_mask": "consensus_mask",
        "comparison::ensemble_uncertainty": "ensemble_uncertainty",
    }
    if key in known:
        return known[key]
    if key.startswith("model_mask::"):
        return f"model_mask_{_safe_export_name(key.split('::', 1)[1], fallback='model')}"
    if key.startswith("model_probability::"):
        return f"model_confidence_{_safe_export_name(key.split('::', 1)[1], fallback='model')}"
    if key.startswith("grid_cell_defects::"):
        return f"grid_cell_defects_{_safe_export_name(key.split('::', 1)[1], fallback='model')}"
    if key.startswith("result_confidence::"):
        parts = key.split("::")
        kind = parts[1] if len(parts) > 1 else "diagnostic"
        model_id = parts[2] if len(parts) > 2 else "model"
        return (
            f"result_confidence_{_safe_export_name(kind, fallback='diagnostic')}_"
            f"{_safe_export_name(model_id, fallback='model')}"
        )
    return _safe_export_name(str(layer_title or key).lower().replace(" ", "_"), fallback="layer")


def _export_worker_count(requested: int | None, total_items: int) -> int:
    if total_items <= 1:
        return 1
    env_value = os.environ.get(EXPORT_WORKER_ENV)
    if env_value:
        try:
            requested = int(env_value)
        except Exception:
            requested = requested
    if requested is None:
        requested = os.cpu_count() or 4
    return max(1, min(int(total_items), EXPORT_MAX_WORKERS, max(1, int(requested))))


def export_result_layer_jpgs(
    build_result: BuildResult,
    destination: Path | str,
    *,
    layer_key: str,
    map_color: tuple[int, int, int] | list[int] | np.ndarray | None = None,
    background_color: tuple[int, int, int] | list[int] | np.ndarray | None = None,
    quality: int = 95,
    max_workers: int | None = None,
    progress_callback=None,
    cancel_check=None,
) -> dict[str, object]:
    """Export one full result layer for every frame as colorized JPG files."""

    destination_path = Path(destination)
    layer_key_text = str(layer_key or "")
    if not layer_key_text:
        raise ValueError("Layer is not selected.")
    records = tuple(build_result.records or ())
    if not records:
        raise ValueError("Nothing to export.")
    layer_folder = destination_path / f"result_layer_{_logical_export_layer_folder_stem(layer_key_text)}"
    layer_folder.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, str]] = []
    errors: list[str] = []
    map_rgb = _rgb_tuple(map_color, fallback=(255, 64, 64))
    background_rgb = _rgb_tuple(background_color, fallback=(128, 128, 128))
    jpeg_quality = max(1, min(100, int(quality)))
    layer_title = layer_key_text
    used_names: set[str] = set()
    total = len(records)
    if progress_callback is not None:
        progress_callback(0, total, "")
    work_items: list[tuple[int, FrameRecord, Path]] = []
    for index, record in enumerate(records, start=1):
        base_name = _safe_export_name(str(record.key or record.display_name), fallback=f"frame_{index:06d}")
        if Path(base_name).suffix.lower() in {".jpg", ".jpeg"}:
            base_name = Path(base_name).stem
        if base_name in used_names:
            base_name = _unique_export_folder_name(base_name, used_names)
        else:
            used_names.add(base_name)
        work_items.append((index, record, layer_folder / f"{base_name}.jpg"))

    def export_one(
        item: tuple[int, FrameRecord, Path],
    ) -> tuple[int, dict[str, str] | None, str | None, str | None, str]:
        index, record, target_path = item
        if cancel_check is not None and cancel_check():
            return index, None, "Export cancelled", None, str(record.display_name or record.key)
        try:
            resolved = _layer_export_array_for_record(build_result, record, layer_key_text)
            if resolved is None:
                return index, None, "layer is unavailable", None, str(record.display_name or record.key)
            array, title = resolved
            rgb = _colorize_layer_map(array, map_color=map_rgb, background_color=background_rgb)
            if not _save_export_rgb_jpg(target_path, rgb, jpeg_quality):
                return index, None, "failed to save JPG", str(title or ""), str(record.display_name or record.key)
            return (
                index,
                {
                    "record_key": str(record.key),
                    "record_name": str(record.display_name),
                    "destination": str(target_path),
                },
                None,
                str(title or ""),
                str(record.display_name or record.key),
            )
        except Exception as error:
            return index, None, str(error), None, str(record.display_name or record.key)

    worker_count = _export_worker_count(max_workers, len(work_items))
    completed = 0
    results_by_index: dict[int, dict[str, str]] = {}
    executor = ThreadPoolExecutor(max_workers=worker_count)
    try:
        pending = {executor.submit(export_one, item) for item in work_items}
        while pending:
            if cancel_check is not None and cancel_check():
                errors.append("Export cancelled")
                for future in pending:
                    future.cancel()
                break
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                if future.cancelled():
                    continue
                index, file_row, error_text, title, frame_name = future.result()
                completed += 1
                if title:
                    layer_title = title
                if file_row is not None:
                    results_by_index[index] = file_row
                elif error_text:
                    errors.append(f"{frame_name}: {error_text}")
                if progress_callback is not None:
                    progress_callback(completed, total, frame_name)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    exported.extend(results_by_index[index] for index in sorted(results_by_index))

    manifest = {
        "layer_key": layer_key_text,
        "layer_title": layer_title,
        "map_color": list(map_rgb),
        "background_color": list(background_rgb),
        "format": "jpg",
        "max_workers": int(worker_count),
        "exported_count": len(exported),
        "skipped_count": len(errors),
        "files": exported,
        "errors": errors,
    }
    manifest_path = layer_folder / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "layer_key": layer_key_text,
        "layer_title": layer_title,
        "exported_count": len(exported),
        "skipped_count": len(errors),
        "destination": str(layer_folder),
        "manifest_path": str(manifest_path),
        "errors": tuple(errors),
        "files": tuple(exported),
    }


def export_result_layers_jpgs(
    build_result: BuildResult,
    destination: Path | str,
    *,
    layer_keys: tuple[str, ...] | list[str],
    map_color: tuple[int, int, int] | list[int] | np.ndarray | None = None,
    background_color: tuple[int, int, int] | list[int] | np.ndarray | None = None,
    quality: int = 95,
    max_workers: int | None = None,
    progress_callback=None,
    cancel_check=None,
) -> dict[str, object]:
    """Export several full result layers, each into its own logical subfolder."""

    keys = tuple(str(key or "") for key in layer_keys if str(key or ""))
    if not keys:
        raise ValueError("Layer is not selected.")
    records_count = len(tuple(build_result.records or ()))
    total_units = max(1, records_count * len(keys))
    layer_results: list[dict[str, object]] = []
    if progress_callback is not None:
        progress_callback(0, total_units, "")
    for layer_index, layer_key in enumerate(keys):
        if cancel_check is not None and cancel_check():
            break
        offset = layer_index * records_count

        def layer_progress(
            current: int, _total: int, frame_name: str, *, base: int = offset, key: str = layer_key
        ) -> None:
            if progress_callback is None:
                return
            progress_callback(
                min(total_units, base + int(current)),
                total_units,
                f"{_logical_export_layer_folder_stem(key)} {frame_name}".strip(),
            )

        result = export_result_layer_jpgs(
            build_result,
            destination,
            layer_key=layer_key,
            map_color=map_color,
            background_color=background_color,
            quality=quality,
            max_workers=max_workers,
            progress_callback=layer_progress,
            cancel_check=cancel_check,
        )
        layer_results.append(result)
    exported_count = sum(int(result.get("exported_count", 0)) for result in layer_results)
    skipped_count = sum(int(result.get("skipped_count", 0)) for result in layer_results)
    return {
        "layer_count": len(layer_results),
        "requested_layer_count": len(keys),
        "exported_count": int(exported_count),
        "skipped_count": int(skipped_count),
        "destination": str(Path(destination)),
        "layers": tuple(layer_results),
    }


def _model_output_with_confidence_bad_areas(
    output_gray: np.ndarray,
    confidence_gray: np.ndarray,
    *,
    alpha_scale: float = 0.78,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    output = np.asarray(output_gray, dtype=np.uint8)
    confidence = np.asarray(confidence_gray, dtype=np.uint8)
    if confidence.shape != output.shape:
        confidence = resize_grayscale_image(confidence, tuple(int(v) for v in output.shape))
    confidence_unit = _prob_from_gray(confidence)
    uncertainty = build_model_uncertainty(confidence_unit)
    bad_intensity = confidence_bad_area_intensity(uncertainty)
    bad_mask = np.clip(np.round(bad_intensity * 255.0), 0.0, 255.0).astype(np.uint8)

    base = np.stack([output, output, output], axis=2).astype(np.float32)
    red = np.zeros_like(base, dtype=np.float32)
    red[..., 0] = 255.0
    red[..., 1] = 32.0
    red[..., 2] = 56.0
    alpha = np.clip(bad_intensity[..., None] * float(alpha_scale), 0.0, 1.0)
    marked = np.clip(base * (1.0 - alpha) + red * alpha, 0.0, 255.0).astype(np.uint8)

    active = bad_intensity > 0.0
    metadata = {
        "bad_area_fraction": float(np.mean(active, dtype=np.float64)) if active.size else 0.0,
        "mean_bad_intensity": float(np.mean(bad_intensity[active], dtype=np.float64)) if np.any(active) else 0.0,
        "max_uncertainty": float(np.max(uncertainty)) if uncertainty.size else 0.0,
        "mean_uncertainty": float(np.mean(uncertainty, dtype=np.float64)) if uncertainty.size else 0.0,
    }
    return marked, bad_mask, metadata


def _model_output_with_internal_bad_areas(
    output_gray: np.ndarray,
    *,
    threshold: float = 0.5,
    alpha_scale: float = 0.78,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    output = np.asarray(output_gray, dtype=np.uint8)
    probability = _prob_from_gray(output)
    support_mask = _mask_from_gray(output, threshold=threshold)
    internal_probability = _internal_confidence_probability_map(
        probability,
        support_mask=support_mask,
        allow_binary_proxy=True,
    )
    uncertainty = build_model_uncertainty(internal_probability)
    bad_intensity = confidence_bad_area_intensity(uncertainty)
    bad_mask = np.clip(np.round(bad_intensity * 255.0), 0.0, 255.0).astype(np.uint8)

    base = np.stack([output, output, output], axis=2).astype(np.float32)
    red = np.zeros_like(base, dtype=np.float32)
    red[..., 0] = 255.0
    red[..., 1] = 160.0
    red[..., 2] = 32.0
    alpha = np.clip(bad_intensity[..., None] * float(alpha_scale), 0.0, 1.0)
    marked = np.clip(base * (1.0 - alpha) + red * alpha, 0.0, 255.0).astype(np.uint8)

    active = bad_intensity > 0.0
    metadata = {
        "bad_area_fraction": float(np.mean(active, dtype=np.float64)) if active.size else 0.0,
        "mean_bad_intensity": float(np.mean(bad_intensity[active], dtype=np.float64)) if np.any(active) else 0.0,
        "max_uncertainty": float(np.max(uncertainty)) if uncertainty.size else 0.0,
        "mean_uncertainty": float(np.mean(uncertainty, dtype=np.float64)) if uncertainty.size else 0.0,
    }
    return marked, bad_mask, metadata


def _save_bad_area_export_pair(
    *,
    marked: np.ndarray,
    bad_mask: np.ndarray,
    metadata: dict[str, float],
    model_id: str,
    output_path: Path,
    target_dir: Path,
    folder_name: str,
    marked_role: str,
    mask_role: str,
    exported: list[dict[str, str]],
    extra_fields: dict[str, str] | None = None,
    file_suffix: str,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_export_name(Path(output_path).stem, fallback="record")
    model_name = _safe_export_name(str(model_id), fallback="model")
    marked_path = target_dir / f"{stem}__{model_name}__{file_suffix}_marked_output.png"
    bad_mask_path = target_dir / f"{stem}__{model_name}__{file_suffix}_bad_areas.png"
    common = {
        "source": str(output_path),
        "folder": folder_name,
        **(extra_fields or {}),
        **{key: f"{value:.8f}" for key, value in metadata.items()},
    }
    if _rgb_array_to_qimage(marked).save(str(marked_path), "PNG"):
        exported.append(
            {
                "role": marked_role,
                "destination": str(marked_path),
                **common,
            }
        )
    if _grayscale_array_to_qimage(bad_mask).save(str(bad_mask_path), "PNG"):
        exported.append(
            {
                "role": mask_role,
                "destination": str(bad_mask_path),
                **common,
            }
        )


def _export_confidence_bad_area_assets(
    *,
    record: FrameRecord,
    model_id: str,
    output_path_text: str,
    confidence_path_text: str,
    destination_path: Path,
    folder_name: str,
    exported: list[dict[str, str]],
) -> None:
    if not output_path_text or not confidence_path_text:
        return
    output_path = Path(str(output_path_text))
    confidence_path = Path(str(confidence_path_text))
    if not output_path.is_file() or not confidence_path.is_file():
        return
    try:
        output_gray = load_grayscale_image(output_path)
        confidence_gray = load_grayscale_image(confidence_path)
        marked, bad_mask, metadata = _model_output_with_confidence_bad_areas(output_gray, confidence_gray)
    except Exception:
        return

    _save_bad_area_export_pair(
        marked=marked,
        bad_mask=bad_mask,
        metadata=metadata,
        model_id=str(model_id),
        output_path=output_path,
        target_dir=destination_path / folder_name,
        folder_name=folder_name,
        marked_role=f"model_output_marked_confidence_bad_areas:{model_id}",
        mask_role=f"model_confidence_bad_area_mask:{model_id}",
        exported=exported,
        extra_fields={"confidence_source": str(confidence_path)},
        file_suffix="confidence",
    )


def _export_internal_bad_area_assets(
    *,
    record: FrameRecord,
    model_id: str,
    output_path_text: str,
    threshold: float,
    destination_path: Path,
    folder_name: str,
    exported: list[dict[str, str]],
) -> None:
    if not output_path_text:
        return
    output_path = Path(str(output_path_text))
    if not output_path.is_file():
        return
    try:
        output_gray = load_grayscale_image(output_path)
        marked, bad_mask, metadata = _model_output_with_internal_bad_areas(
            output_gray,
            threshold=float(threshold),
        )
    except Exception:
        return
    _save_bad_area_export_pair(
        marked=marked,
        bad_mask=bad_mask,
        metadata=metadata,
        model_id=str(model_id),
        output_path=output_path,
        target_dir=destination_path / folder_name,
        folder_name=folder_name,
        marked_role=f"model_output_marked_internal_bad_areas:{model_id}",
        mask_role=f"model_internal_bad_area_mask:{model_id}",
        exported=exported,
        extra_fields={"internal_confidence_source": "model_output_probability_proxy"},
        file_suffix="internal",
    )


def export_record_assets(
    build_result: BuildResult,
    record: FrameRecord,
    destination: Path | str,
    *,
    write_manifest: bool = True,
) -> dict[str, object]:
    """Export all files attached to one matrix record into source-named folders."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    model_specs = {str(spec.model_id): spec for spec in build_result.model_specs}
    used_folder_names: set[str] = set()
    exported: list[dict[str, str]] = []

    def copy_source(
        source_path_text: str | None, source_folder: Path | None, fallback_folder_name: str, role: str
    ) -> None:
        if not source_path_text:
            return
        source_path = Path(str(source_path_text))
        if not source_path.is_file():
            return
        folder_name = _unique_export_folder_name(
            (source_folder.name if source_folder is not None else source_path.parent.name) or fallback_folder_name,
            used_folder_names,
        )
        target_dir = destination_path / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        exported.append(
            {
                "role": str(role),
                "source": str(source_path),
                "destination": str(target_path),
                "folder": folder_name,
            }
        )

    original_folder = build_result.original_folder.path if build_result.original_folder is not None else None
    copy_source(record.original_path or record.base_path, original_folder, "original", "original")

    for model_id, path_text in (record.model_mask_paths or {}).items():
        spec = model_specs.get(str(model_id))
        copy_source(
            path_text,
            spec.mask_folder if spec is not None else None,
            f"model_{model_id}",
            f"model_output:{model_id}",
        )
        if path_text:
            folder_name = _unique_export_folder_name(
                f"marked_internal_bad_areas_{spec.display_name if spec is not None else model_id}",
                used_folder_names,
            )
            _export_internal_bad_area_assets(
                record=record,
                model_id=str(model_id),
                output_path_text=str(path_text),
                threshold=float(spec.threshold if spec is not None else 0.5),
                destination_path=destination_path,
                folder_name=folder_name,
                exported=exported,
            )
    for model_id, path_text in (record.model_prob_paths or {}).items():
        if not path_text:
            continue
        spec = model_specs.get(str(model_id))
        copy_source(
            path_text,
            spec.prob_folder if spec is not None else None,
            f"confidence_{model_id}",
            f"model_confidence:{model_id}",
        )
        output_path_text = str((record.model_mask_paths or {}).get(str(model_id)) or "")
        if output_path_text:
            folder_name = _unique_export_folder_name(
                f"marked_confidence_bad_areas_{spec.display_name if spec is not None else model_id}",
                used_folder_names,
            )
            _export_confidence_bad_area_assets(
                record=record,
                model_id=str(model_id),
                output_path_text=output_path_text,
                confidence_path_text=str(path_text),
                destination_path=destination_path,
                folder_name=folder_name,
                exported=exported,
            )

    manifest = {
        "record_key": str(record.key),
        "record_name": str(record.display_name),
        "exported_count": len(exported),
        "files": exported,
    }
    manifest_path = None
    if write_manifest:
        manifest_path = destination_path / f"{_safe_export_name(record.key, fallback='record')}_export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "record_key": str(record.key),
        "exported_count": len(exported),
        "destination": str(destination_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "files": tuple(exported),
    }


def export_ranked_frames(
    build_result: BuildResult,
    destination: Path | str,
    *,
    top_k: int,
    neighbor_radius: int = 1,
    metric_key: str = "export_priority_score",
    selection_mode: str = EXPORT_SELECTION_MODE_COUNT,
    top_percent: float = 10.0,
    percentile_threshold: float = 90.0,
) -> dict[str, object]:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    records = [record for record in build_result.records if record.summary is not None]
    if not records:
        raise ValueError("Nothing to export. Compute analytics first.")
    selected = list(
        select_candidate_records(
            replace(build_result, records=tuple(records)),
            metric_key=metric_key,
            selection_mode=selection_mode,
            top_k=top_k,
            top_percent=top_percent,
            percentile_threshold=percentile_threshold,
        )
    )
    percentile_map = compute_metric_percentiles(records, metric_key)
    sequence_groups = _sequence_groups(tuple(build_result.records))
    selected_keys = {record.key for record in selected}
    supplemental_keys: set[str] = set()
    radius = max(0, int(neighbor_radius))
    if radius > 0:
        for group in sequence_groups.values():
            key_to_index = {item.key: index for index, item in enumerate(group)}
            for record in selected:
                if record.key not in key_to_index:
                    continue
                center = key_to_index[record.key]
                for offset in range(-radius, radius + 1):
                    neighbor_index = center + offset
                    if neighbor_index < 0 or neighbor_index >= len(group):
                        continue
                    candidate = group[neighbor_index]
                    if candidate.key not in selected_keys:
                        supplemental_keys.add(candidate.key)

    export_records = {
        record.key: record for record in build_result.records if record.key in selected_keys | supplemental_keys
    }
    manifest: dict[str, object] = {
        "metric_key": metric_key,
        "selection_mode": selection_mode,
        "top_k": int(top_k),
        "top_percent": float(top_percent),
        "percentile_threshold": float(percentile_threshold),
        "selected_keys": [record.key for record in selected],
        "supplemental_keys": sorted(supplemental_keys, key=natural_sort_key),
        "scores": {record.key: metric_value_for_record(record, metric_key) for record in export_records.values()},
        "score_percentiles": {
            record.key: float(percentile_map.get(record.key, 0.0)) for record in export_records.values()
        },
        "reasons": {
            record.key: ([metric_key, "primary"] if record.key in selected_keys else [metric_key, "neighbor"])
            for record in export_records.values()
        },
    }

    original_dir = destination_path / "original"
    masks_root = destination_path / "models"
    probs_root = destination_path / "probabilities"
    for root in (original_dir, masks_root, probs_root):
        root.mkdir(parents=True, exist_ok=True)

    for key, record in export_records.items():
        normalized_name = key.replace("/", "__")
        if record.original_path:
            shutil.copy2(record.original_path, original_dir / Path(normalized_name).name)
        for model_id, path_text in record.model_mask_paths.items():
            model_dir = masks_root / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_text, model_dir / Path(normalized_name).name)
        for model_id, path_text in record.model_prob_paths.items():
            if not path_text:
                continue
            model_dir = probs_root / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_text, model_dir / Path(normalized_name).name)

    manifest_path = destination_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "selected_count": len(selected_keys),
        "supplemental_count": len(supplemental_keys),
        "manifest_path": str(manifest_path),
        "selected_keys": manifest["selected_keys"],
        "supplemental_keys": manifest["supplemental_keys"],
    }
