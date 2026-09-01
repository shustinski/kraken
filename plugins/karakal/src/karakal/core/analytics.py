"""Frame collection and analytics orchestration."""

from __future__ import annotations

from .analysis_cache import (
    _load_cached_record_payload,
    _record_payload_cache_key,
    _store_cached_record_payload,
)

from .confidence_analysis import (
    _mask_from_gray,
    _model_output_confidence_metrics,
    _point_internal_confidence,
    _polygon_frame_confidence,
    _prob_from_gray,
)

from .image_io import (
    _FolderPathLookup,
    _build_folder_path_lookup,
    _clear_runtime_image_caches,
    _load_optional_gray,
    _resolve_aux_path_from_lookup,
    _resolve_model_path_for_key,
    build_folder_index,
    build_frame_identity,
    build_prediction_view_from_gray,
    load_grayscale_image,
    natural_sort_key,
    resize_grayscale_image,
)

from .mask_metrics import (
    _aggregate_inter_model_point_scores,
    _aggregate_inter_model_polygon_scores,
    _configured_combined_pair_metric_values,
    _configured_confidence_pair_metric_values,
    _configured_pair_metric_values,
    _infer_geometry_mode,
    _pairwise_model_comparisons,
    _point_diagnostic_metrics,
    _polygon_bce_score,
    _symmetric_binary_cross_entropy,
    compute_comparison_score,
)

from .mask_primitives import (
    _clip01,
    _has_fast_component_label_backend,
    _mask_structure,
)

from .metric_keys import (
    _available_metric_keys_for_models,
    _model_metric_key,
    _normalized_comparison_pairs,
    _parse_model_metric_key,
    _record_metric_value,
    compute_metric_percentiles,
    metric_higher_is_better,
    parse_combined_pair_metric_key,
    parse_confidence_pair_metric_key,
    parse_pair_metric_key,
)

from .repository_shared import (
    ANALYTICS_BATCH_TARGETS_PER_WORKER,
    ANALYTICS_MAX_BATCH_SIZE,
    ANALYTICS_MEMORY_FRACTION,
    ANALYTICS_STALL_PROGRESS_SECONDS,
    ANALYTICS_STALL_TIMEOUT_SECONDS,
    ANALYTICS_WAIT_TIMEOUT_SECONDS,
    ANALYTICS_WORKER_ENV,
    BuildCancelledError,
    BuildOptions,
    BuildResult,
    ComparisonMode,
    ComparisonPairSelection,
    ComparisonTarget,
    EPS,
    EnsembleComparisonRequest,
    FIRST_COMPLETED,
    FolderSpec,
    FrameAnalysisSummary,
    FrameComparisonResult,
    FrameRecord,
    GeometryMode,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    ModelDiagnosticMetrics,
    ModelFrameResult,
    ModelOutputConfidenceMetrics,
    ModelSpec,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    PairwiseComparisonRequest,
    Path,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    WINDOWS_PROCESS_ANALYTICS_MAX_WORKERS,
    WINDOWS_THREAD_ANALYTICS_MAX_WORKERS,
    WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS,
    _LAST_ANALYTICS_WORKER_PLAN,
    _LOGGER,
    compare_ensemble,
    compare_pairwise,
    ctypes,
    current_profiler,
    math,
    np,
    os,
    perf_counter,
    profile_stage,
    replace,
    wait,
)


def _analytics_batch_size(total_items: int, worker_count: int) -> int:
    total = max(1, int(total_items))
    workers = max(1, int(worker_count))
    target_batches = max(1, workers * ANALYTICS_BATCH_TARGETS_PER_WORKER)
    return max(1, min(ANALYTICS_MAX_BATCH_SIZE, math.ceil(total / target_batches)))


def _use_thread_pool_for_analytics() -> bool:
    # ProcessPoolExecutor is fragile inside a PyQt worker thread on Windows:
    # child process spawn/import or pickle stalls look like a frozen progress bar.
    return os.name == "nt"


def _analytics_worker_env_override() -> int | None:
    value = os.environ.get(ANALYTICS_WORKER_ENV)
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as error:
        _LOGGER.warning("Ignoring invalid %s value %r: %s", ANALYTICS_WORKER_ENV, value, error)
        return None
    return parsed if parsed > 0 else None


def _available_memory_bytes() -> int | None:
    if os.name == "nt":

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError, ValueError) as error:
            _LOGGER.debug("Windows memory availability is unavailable: %s", error)
            return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages > 0:
            return page_size * available_pages
    except (AttributeError, OSError, TypeError, ValueError) as error:
        _LOGGER.debug("POSIX memory availability is unavailable: %s", error)
        return None
    return None


def _estimated_analytics_worker_memory_bytes(
    *,
    analysis_max_side: int | None,
    model_count: int,
    include_model_confidence: bool,
    include_model_output_confidence: bool,
    include_pairwise_metrics: bool,
) -> int:
    side = int(analysis_max_side or 2048)
    side = max(64, min(side, 8192))
    pixels = int(side) * int(side)
    models = max(1, int(model_count))
    layer_factor = 4 + min(models, 8)
    if include_pairwise_metrics:
        layer_factor += min(models * 2, 12)
    if include_model_confidence:
        layer_factor += min(models * 3, 16)
    if include_model_output_confidence:
        layer_factor += min(models, 8)
    return max(64 * 1024 * 1024, int(pixels * layer_factor * 4))


def _memory_limited_worker_count(
    *,
    analysis_max_side: int | None,
    model_count: int,
    include_model_confidence: bool,
    include_model_output_confidence: bool,
    include_pairwise_metrics: bool,
) -> int | None:
    available = _available_memory_bytes()
    if available is None or available <= 0:
        return None
    per_worker = _estimated_analytics_worker_memory_bytes(
        analysis_max_side=analysis_max_side,
        model_count=model_count,
        include_model_confidence=include_model_confidence,
        include_model_output_confidence=include_model_output_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
    )
    budget = max(per_worker, int(float(available) * ANALYTICS_MEMORY_FRACTION))
    return max(1, int(budget // max(1, per_worker)))


def _cpu_limited_thread_worker_count(
    *,
    cpu_limit: int,
    analysis_max_side: int | None,
    include_model_confidence: bool,
) -> int:
    side_limit = int(analysis_max_side or 0)
    if include_model_confidence:
        if side_limit == 0 or side_limit > 1024:
            return min(cpu_limit, WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS)
        return min(cpu_limit, max(WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS, cpu_limit // 2))
    if side_limit > 0 and side_limit <= 512:
        return min(cpu_limit, max(16, cpu_limit))
    if side_limit > 0 and side_limit <= 1024:
        return min(cpu_limit, WINDOWS_THREAD_ANALYTICS_MAX_WORKERS)
    return min(cpu_limit, max(16, WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS))


def _analytics_worker_count(
    max_workers: int,
    *,
    geometry_mode: GeometryMode,
    analysis_max_side: int | None,
    include_model_confidence: bool,
    include_model_output_confidence: bool = False,
    include_pairwise_metrics: bool = True,
    model_count: int = 1,
    use_thread_pool: bool,
) -> int:
    requested = max(1, int(max_workers))
    override = _analytics_worker_env_override()
    if override is not None:
        return min(requested, max(1, int(override)))
    if os.name != "nt" or geometry_mode == GeometryMode.POINT:
        return requested
    cpu_limit = max(1, os.cpu_count() or requested)
    if use_thread_pool:
        safe_limit = _cpu_limited_thread_worker_count(
            cpu_limit=cpu_limit,
            analysis_max_side=analysis_max_side,
            include_model_confidence=include_model_confidence,
        )
        memory_limit = _memory_limited_worker_count(
            analysis_max_side=analysis_max_side,
            model_count=model_count,
            include_model_confidence=include_model_confidence,
            include_model_output_confidence=include_model_output_confidence,
            include_pairwise_metrics=include_pairwise_metrics,
        )
        if memory_limit is not None:
            safe_limit = min(safe_limit, memory_limit)
        return min(requested, max(1, safe_limit))
    safe_limit = min(cpu_limit, WINDOWS_PROCESS_ANALYTICS_MAX_WORKERS)
    side_limit = int(analysis_max_side or 0)
    if include_model_confidence and side_limit > 0 and side_limit <= 768:
        safe_limit = min(cpu_limit, 5)
    return min(requested, max(1, safe_limit))


def _remember_analytics_worker_plan(
    *,
    requested: int,
    selected: int,
    use_thread_pool: bool,
    geometry_mode: GeometryMode,
    analysis_max_side: int | None,
    model_count: int,
    include_model_confidence: bool,
    include_model_output_confidence: bool,
    include_pairwise_metrics: bool,
) -> None:
    _LAST_ANALYTICS_WORKER_PLAN.clear()
    _LAST_ANALYTICS_WORKER_PLAN.update(
        {
            "requested": int(requested),
            "selected": int(selected),
            "use_thread_pool": bool(use_thread_pool),
            "geometry_mode": str(getattr(geometry_mode, "value", geometry_mode)),
            "analysis_max_side": None if analysis_max_side is None else int(analysis_max_side),
            "model_count": int(model_count),
            "include_model_confidence": bool(include_model_confidence),
            "include_model_output_confidence": bool(include_model_output_confidence),
            "include_pairwise_metrics": bool(include_pairwise_metrics),
        }
    )


def last_analytics_worker_plan() -> dict[str, object]:
    return dict(_LAST_ANALYTICS_WORKER_PLAN)


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
    include_model_diagnostics: bool = True,
    include_structure_details: bool = True,
    include_model_output_probabilities: bool = True,
    include_source_grays: bool = True,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, bool],
    np.ndarray | None,
    GeometryMode,
    dict[str, object],
]:
    include_pairwise_metrics = bool(include_pairwise_metrics)
    include_model_diagnostics = bool(include_model_diagnostics)
    include_model_confidence = bool(include_model_confidence)
    include_model_output_probabilities = bool(include_model_output_probabilities)
    include_source_grays = bool(include_source_grays)
    include_mask_structures = include_pairwise_metrics or include_model_diagnostics
    include_boundary_distance = include_pairwise_metrics

    original_gray = (
        _load_optional_gray(record.original_path, max_side=analysis_max_side) if include_original_gray else None
    )
    probabilities: dict[str, np.ndarray] = {}
    output_probabilities: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    source_grays: dict[str, np.ndarray] = {}
    model_structures: dict[str, dict[str, object]] = {}
    model_diagnostics: dict[str, object] = {}
    model_confidence: dict[str, object] = {}
    model_confidence_output_available: dict[str, bool] = {}
    model_views: dict[str, object] = {}

    target_shape = tuple(int(v) for v in original_gray.shape) if original_gray is not None else None

    loaded_rows: list[tuple[ModelSpec, np.ndarray, np.ndarray, bool]] = []
    first_point_candidate = None
    for spec in model_specs:
        mask_gray = _load_optional_gray(
            record.model_mask_paths.get(spec.model_id), target_shape=target_shape, max_side=analysis_max_side
        )
        if mask_gray is None:
            continue
        if target_shape is None:
            target_shape = tuple(int(v) for v in mask_gray.shape)
        if original_gray is not None and tuple(int(v) for v in original_gray.shape) != target_shape:
            original_gray = _load_optional_gray(
                record.original_path, target_shape=target_shape, max_side=analysis_max_side
            )
        prob_gray = (
            _load_optional_gray(
                record.model_prob_paths.get(spec.model_id), target_shape=target_shape, max_side=analysis_max_side
            )
            if include_model_output_probabilities
            else None
        )
        uses_binary_probability_proxy = prob_gray is None
        model_confidence_output_available[spec.model_id] = not uses_binary_probability_proxy
        if prob_gray is None:
            prob_gray = mask_gray
        loaded_rows.append((spec, mask_gray, prob_gray, uses_binary_probability_proxy))
        if geometry_mode == GeometryMode.AUTO and first_point_candidate is None:
            first_point_candidate = build_prediction_view_from_gray(
                spec.display_name,
                np.asarray(mask_gray, dtype=np.uint8),
                threshold=int(round(float(spec.threshold) * 255.0)),
            )

    resolved_geometry_mode = geometry_mode
    if resolved_geometry_mode == GeometryMode.AUTO:
        resolved_geometry_mode = (
            _infer_geometry_mode(first_point_candidate) if first_point_candidate is not None else GeometryMode.MASK
        )

    for spec, mask_gray, prob_gray, uses_binary_probability_proxy in loaded_rows:
        prob_map = _prob_from_gray(mask_gray)
        mask = _mask_from_gray(mask_gray, threshold=spec.threshold)
        if include_source_grays:
            source_grays[spec.model_id] = (np.asarray(mask_gray, dtype=np.float32) / 255.0).astype(
                np.float32, copy=False
            )
        probabilities[spec.model_id] = prob_map.astype(np.float32)
        if include_model_output_probabilities:
            output_prob_map = _prob_from_gray(prob_gray)
            output_probabilities[spec.model_id] = output_prob_map.astype(np.float32)
        masks[spec.model_id] = mask.astype(bool)

        if resolved_geometry_mode == GeometryMode.POINT:
            need_prediction_view = (
                include_pairwise_metrics
                or include_model_confidence
                or include_model_diagnostics
            )
            prediction_view = None
            if need_prediction_view:
                prediction_view = build_prediction_view_from_gray(
                    spec.display_name,
                    np.asarray(mask_gray, dtype=np.uint8),
                    threshold=int(round(float(spec.threshold) * 255.0)),
                )
            if prediction_view is not None and include_pairwise_metrics:
                model_views[spec.model_id] = prediction_view
            diagnostics = (
                _point_diagnostic_metrics(prediction_view)
                if (prediction_view is not None and include_model_diagnostics)
                else None
            )
            if diagnostics is not None:
                model_diagnostics[spec.model_id] = diagnostics
            if include_model_confidence and prediction_view is not None:
                model_confidence[spec.model_id] = _point_internal_confidence(
                    prediction_view,
                    neighborhood_radius=int(point_confidence_radius),
                    include_objects=include_confidence_objects,
                )
            if diagnostics is not None and prediction_view is not None:
                model_structures[spec.model_id] = {
                    "component_count": int(diagnostics.point_count),
                    "area_fraction": float(
                        max(0.0, min(1.0, diagnostics.point_count / max(1.0, float(prediction_view.pred_gray.size))))
                    ),
                    "skeleton_length": float(diagnostics.mean_radius),
                }
            continue

        mask_structure = None
        if include_mask_structures:
            mask_structure = _mask_structure(
                mask,
                include_skeleton=include_structure_details,
                include_boundary_distance=include_boundary_distance,
                include_component_labels=bool(
                    include_model_diagnostics or _has_fast_component_label_backend()
                ),
            )
            model_structures[spec.model_id] = mask_structure
        if include_model_diagnostics and mask_structure is not None:
            area_fraction = float(mask_structure["area_fraction"])
            proxy_score = float(_clip01(1.0 - area_fraction))
            model_diagnostics[spec.model_id] = ModelDiagnosticMetrics(
                area_fraction=area_fraction,
                component_count=int(mask_structure["component_count"]),
                skeleton_length=float(mask_structure["skeleton_length"]),
                proxy_score=proxy_score,
            )
        if include_model_confidence:
            model_confidence[spec.model_id] = _polygon_frame_confidence(
                prob_map,
                mask,
                uncertainty_delta=float(confidence_uncertainty_delta),
                summary_metric=str(polygon_confidence_summary),
                allow_binary_proxy=True,
            )
    return (
        probabilities,
        output_probabilities,
        masks,
        source_grays,
        model_structures,
        model_diagnostics,
        model_confidence,
        model_confidence_output_available,
        original_gray,
        resolved_geometry_mode,
        model_views,
    )


def _metric_requires_model_confidence(metric_key: str | None) -> bool:
    parsed = _parse_model_metric_key(str(metric_key or ""))
    if parsed is None:
        return False
    family, _model_id = parsed
    return family in {"model_confidence", "model_uncertain_fraction", "model_point_contrast"}


def _build_frame_comparison_result(
    *,
    frame_id: str,
    model_specs: tuple[ModelSpec, ...],
    probabilities_by_model: dict[str, np.ndarray],
    masks_by_model: dict[str, np.ndarray],
    geometry_mode: GeometryMode,
    threshold: float,
    consensus_threshold: float,
    connectivity: int,
    pruning_min_length_px: int = 5,
    compute_level: str = "standard",
) -> FrameComparisonResult | None:
    model_frames: list[ModelFrameResult] = []
    profile = "point" if geometry_mode == GeometryMode.POINT else "polygon"
    for spec in model_specs:
        mask = masks_by_model.get(spec.model_id)
        if mask is None:
            continue
        model_frames.append(
            ModelFrameResult(
                model_id=str(spec.model_id),
                frame_id=str(frame_id),
                probability_map=probabilities_by_model.get(spec.model_id),
                binary_mask=np.asarray(mask, dtype=bool),
                metadata={
                    "geometry_mode": profile,
                    "threshold": float(getattr(spec, "threshold", threshold) or threshold),
                },
            )
        )
    if len(model_frames) < 2:
        return None
    if len(model_frames) == 2:
        return compare_pairwise(
            PairwiseComparisonRequest(
                frame_id=str(frame_id),
                model_a=model_frames[0],
                model_b=model_frames[1],
                profile=profile,
                threshold=float(threshold),
                connectivity=int(connectivity),
                pruning_min_length_px=int(pruning_min_length_px),
                evidence_provider_version="karakal-detail",
                compute_level=str(compute_level or "standard"),
            )
        ).frame
    return compare_ensemble(
        EnsembleComparisonRequest(
            frame_id=str(frame_id),
            models=tuple(model_frames),
            profile="mixed" if geometry_mode != GeometryMode.POINT else "point",
            threshold=float(threshold),
            consensus_threshold=float(consensus_threshold),
            connectivity=int(connectivity),
            pruning_min_length_px=int(pruning_min_length_px),
            evidence_provider_version="karakal-detail",
            compute_level=str(compute_level or "standard"),
        )
    ).frame


def _comparison_metric_values(result: FrameComparisonResult | None) -> dict[str, float]:
    if result is None:
        return {}
    values: dict[str, float] = {}
    for key, value in result.risk.items():
        if value is not None and math.isfinite(float(value)):
            values[f"comparison_risk_{key}"] = float(value)
    for metric in result.metrics:
        if not metric.valid or metric.value is None:
            continue
        try:
            numeric = float(metric.value)
        except (TypeError, ValueError) as error:
            _LOGGER.warning("Ignoring non-numeric comparison metric %s=%r: %s", metric.name, metric.value, error)
            continue
        if math.isfinite(numeric):
            values[f"comparison::{metric.name}"] = numeric
    return values


CONFIDENCE_COMPARISON_METRIC_KEYS = frozenset(
    {
        "confidence_model_score",
        "confidence_difference_score",
        "confidence_bce_score",
        "confidence_threshold_crossing_score",
    }
)


def _confidence_pairwise_metric_values(output_probabilities_by_model: dict[str, np.ndarray]) -> dict[str, float]:
    items = [
        (str(model_id), np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0))
        for model_id, probability in (output_probabilities_by_model or {}).items()
        if probability is not None
    ]
    if len(items) < 2:
        return {}
    diff_scores: list[float] = []
    bce_scores: list[float] = []
    crossing_scores: list[float] = []
    for first_index in range(len(items)):
        _first_id, first = items[first_index]
        for second_index in range(first_index + 1, len(items)):
            _second_id, second = items[second_index]
            if first.shape != second.shape or first.size == 0:
                continue
            abs_diff = np.abs(first - second)
            diff_score = 100.0 * (1.0 - float(np.mean(abs_diff, dtype=np.float64)))
            bce_value = _symmetric_binary_cross_entropy(first, second)
            bce_score = _polygon_bce_score(bce_value)
            threshold_crossing = np.asarray(first >= 0.5, dtype=bool) ^ np.asarray(second >= 0.5, dtype=bool)
            crossing_score = 100.0 * (1.0 - float(np.mean(threshold_crossing, dtype=np.float64)))
            diff_scores.append(float(_clip01(diff_score / 100.0) * 100.0))
            bce_scores.append(float(_clip01(bce_score / 100.0) * 100.0))
            crossing_scores.append(float(_clip01(crossing_score / 100.0) * 100.0))
    if not diff_scores:
        return {}
    diff_mean = float(np.mean(np.asarray(diff_scores, dtype=np.float64), dtype=np.float64))
    bce_mean = float(np.mean(np.asarray(bce_scores, dtype=np.float64), dtype=np.float64))
    crossing_mean = float(np.mean(np.asarray(crossing_scores, dtype=np.float64), dtype=np.float64))
    return {
        "confidence_difference_score": diff_mean,
        "confidence_bce_score": bce_mean,
        "confidence_threshold_crossing_score": crossing_mean,
        "confidence_model_score": float(
            np.mean(np.asarray([diff_mean, bce_mean, crossing_mean], dtype=np.float64), dtype=np.float64)
        ),
    }


def _metric_requires_model_output_confidence(metric_key: str | None) -> bool:
    if str(metric_key or "") in CONFIDENCE_COMPARISON_METRIC_KEYS:
        return True
    if (
        parse_confidence_pair_metric_key(metric_key) is not None
        or parse_combined_pair_metric_key(metric_key) is not None
    ):
        return True
    parsed = _parse_model_metric_key(str(metric_key or ""))
    if parsed is None:
        return False
    family, _model_id = parsed
    return family == "model_output_confidence"


def _metric_requires_pairwise_metrics(metric_key: str | None) -> bool:
    metric_key_text = str(metric_key or "overall_frame_score")
    if (
        parse_pair_metric_key(metric_key_text) is not None
        or parse_combined_pair_metric_key(metric_key_text) is not None
    ):
        return True
    parsed = _parse_model_metric_key(metric_key_text)
    if parsed is not None:
        family, _model_id = parsed
        return family not in {
            "model_confidence",
            "model_uncertain_fraction",
            "model_point_contrast",
            "model_output_confidence",
        }
    return metric_key_text in {
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
    }


def _analyze_record_payload(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    cache_enabled: bool,
    include_model_confidence: bool = True,
    include_model_output_confidence: bool = False,
    include_pairwise_metrics: bool = True,
    comparison_pairs: tuple[ComparisonPairSelection, ...] = (),
) -> dict[str, object] | None:
    timings_ms: dict[str, float] = {}
    cache_key = None
    if cache_enabled:
        cache_key = _record_payload_cache_key(
            record,
            model_specs,
            analysis_max_side,
            geometry_mode,
            point_match_radius,
            boundary_radius,
            confidence_uncertainty_delta,
            point_confidence_radius,
            polygon_confidence_summary,
            include_model_confidence=bool(include_model_confidence),
            include_model_output_confidence=bool(include_model_output_confidence),
            include_pairwise_metrics=bool(include_pairwise_metrics),
            comparison_pairs=comparison_pairs,
        )
        with profile_stage("validation.cache.record.read", frame_id=record.key):
            cached = _load_cached_record_payload(cache_key)
        if cached is not None:
            profiler = current_profiler()
            if profiler is not None:
                profiler.increment("cache.disk.hits")
            return cached
        profiler = current_profiler()
        if profiler is not None:
            profiler.increment("cache.disk.misses")

    include_model_diagnostics = bool(
        include_model_confidence or (include_pairwise_metrics and geometry_mode != GeometryMode.MASK)
    )

    load_started = perf_counter()
    with profile_stage("validation.frame.loading_preprocess", frame_id=record.key):
        (
            probabilities_by_model,
            output_probabilities_by_model,
            masks_by_model,
            _source_grays_by_model,
            model_structures,
            model_diagnostics,
            model_confidence,
            model_confidence_output_available,
            _original_gray,
            resolved_geometry_mode,
            model_views,
        ) = _build_model_payloads(
            record,
            model_specs,
            analysis_max_side=analysis_max_side,
            geometry_mode=geometry_mode,
            point_match_radius=point_match_radius,
            boundary_radius=boundary_radius,
            confidence_uncertainty_delta=confidence_uncertainty_delta,
            point_confidence_radius=point_confidence_radius,
            polygon_confidence_summary=polygon_confidence_summary,
            include_confidence_objects=False,
            include_original_gray=False,
            include_model_confidence=include_model_confidence,
            include_pairwise_metrics=include_pairwise_metrics,
            include_model_diagnostics=include_model_diagnostics,
            include_structure_details=False,
            include_model_output_probabilities=include_model_output_confidence,
            include_source_grays=False,
        )
    timings_ms["loading_preprocess"] = 1000.0 * (perf_counter() - load_started)
    probabilities = list(probabilities_by_model.values())
    if not probabilities:
        return None

    metrics_started = perf_counter()
    pairwise_rows: tuple[dict[str, object], ...] = ()
    disagreement = 0.0
    model_model_score = 1.0
    if include_pairwise_metrics:
        with profile_stage("validation.metrics.pairwise", frame_id=record.key):
            pairwise_rows = _pairwise_model_comparisons(
                probabilities_by_model,
                masks_by_model,
                geometry_mode=resolved_geometry_mode,
                model_views=model_views,
                model_structures=model_structures,
                point_match_radius=point_match_radius,
            )
        agreement_scores = np.asarray(
            [float(row.get("agreement_score", 0.0)) for row in pairwise_rows], dtype=np.float64
        )
        disagreement = float(np.mean(1.0 - agreement_scores, dtype=np.float64)) if agreement_scores.size else 0.0
        model_model_score = float(np.mean(agreement_scores, dtype=np.float64)) if agreement_scores.size else 1.0
    pair_metric_values = _configured_pair_metric_values(
        _normalized_comparison_pairs(comparison_pairs),
        pairwise_rows,
        masks_by_model,
    )
    confidence_pair_metric_values = (
        _configured_confidence_pair_metric_values(
            _normalized_comparison_pairs(comparison_pairs),
            {
                str(model_id): probability
                for model_id, probability in output_probabilities_by_model.items()
                if bool(model_confidence_output_available.get(str(model_id), False))
            },
        )
        if include_model_output_confidence
        else {}
    )
    combined_pair_metric_values = _configured_combined_pair_metric_values(
        _normalized_comparison_pairs(comparison_pairs),
        pair_metric_values,
        confidence_pair_metric_values,
    )

    vector: dict[str, float] = {}

    model_confidence_output: dict[str, ModelOutputConfidenceMetrics] = {}
    confidence_pairwise_metrics = (
        _confidence_pairwise_metric_values(
            {
                str(model_id): probability
                for model_id, probability in output_probabilities_by_model.items()
                if bool(model_confidence_output_available.get(str(model_id), False))
            }
        )
        if include_model_output_confidence
        else {}
    )
    if include_model_output_confidence:
        with profile_stage("validation.metrics.confidence", frame_id=record.key):
            for model_id, probability in output_probabilities_by_model.items():
                if not bool(model_confidence_output_available.get(str(model_id), False)):
                    continue
                model_confidence_output[str(model_id)] = _model_output_confidence_metrics(probability)
    timings_ms["metrics"] = 1000.0 * (perf_counter() - metrics_started)

    assemble_started = perf_counter()
    payload = {
        "key": record.key,
        "geometry_mode": resolved_geometry_mode.value,
        "vector": vector,
        "disagreement": float(disagreement),
        "model_model_score": float(_clip01(model_model_score)),
        "model_diagnostics": model_diagnostics,
        "model_confidence": model_confidence,
        "model_confidence_output": model_confidence_output,
        "confidence_pairwise_metrics": confidence_pairwise_metrics,
        "confidence_pair_metric_values": confidence_pair_metric_values,
        "combined_pair_metric_values": combined_pair_metric_values,
        "pair_metric_values": pair_metric_values,
        "model_confidence_output_available": model_confidence_output_available,
        "pairwise_rows": pairwise_rows,
        "timings_ms": timings_ms,
    }
    timings_ms["payload_assembly"] = 1000.0 * (perf_counter() - assemble_started)
    if cache_enabled and cache_key is not None:
        with profile_stage("validation.cache.record.write", frame_id=record.key):
            _store_cached_record_payload(cache_key, payload)
    return payload


def _analyze_record_payload_for_executor(
    args: tuple[
        FrameRecord,
        tuple[ModelSpec, ...],
        int | None,
        GeometryMode,
        float,
        int,
        float,
        int,
        str,
        bool,
        bool,
        bool,
        bool,
        tuple[ComparisonPairSelection, ...],
    ],
) -> tuple[str, dict[str, object] | None]:
    (
        record,
        model_specs,
        analysis_max_side,
        geometry_mode,
        point_match_radius,
        boundary_radius,
        confidence_uncertainty_delta,
        point_confidence_radius,
        polygon_confidence_summary,
        cache_enabled,
        include_model_confidence,
        include_model_output_confidence,
        include_pairwise_metrics,
        comparison_pairs,
    ) = args
    return record.key, _analyze_record_payload(
        record,
        model_specs,
        analysis_max_side,
        geometry_mode,
        point_match_radius,
        boundary_radius,
        confidence_uncertainty_delta,
        point_confidence_radius,
        polygon_confidence_summary,
        cache_enabled,
        include_model_confidence,
        include_model_output_confidence,
        include_pairwise_metrics,
        comparison_pairs,
    )


def _analyze_record_payload_batch_for_executor(
    batch_args: tuple[
        tuple[
            FrameRecord,
            tuple[ModelSpec, ...],
            int | None,
            GeometryMode,
            float,
            int,
            float,
            int,
            str,
            bool,
            bool,
            bool,
            bool,
            tuple[ComparisonPairSelection, ...],
        ],
        ...,
    ],
) -> tuple[tuple[str, dict[str, object] | None], ...]:
    results: list[tuple[str, dict[str, object] | None]] = []
    for args in batch_args:
        results.append(_analyze_record_payload_for_executor(args))
    _clear_runtime_image_caches()
    return tuple(results)


def _iter_record_payloads(
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
    include_model_output_confidence: bool = False,
    include_pairwise_metrics: bool = True,
    comparison_pairs: tuple[ComparisonPairSelection, ...] = (),
    progress_callback=None,
    state_callback=None,
    cancel_check=None,
):
    """Yield per-record analytics payloads without retaining the whole dataset in RAM."""

    use_thread_pool = _use_thread_pool_for_analytics()
    worker_count = _analytics_worker_count(
        max_workers,
        geometry_mode=geometry_mode,
        analysis_max_side=analysis_max_side,
        include_model_confidence=include_model_confidence,
        include_model_output_confidence=include_model_output_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
        model_count=len(model_specs),
        use_thread_pool=use_thread_pool,
    )
    _remember_analytics_worker_plan(
        requested=max_workers,
        selected=worker_count,
        use_thread_pool=use_thread_pool,
        geometry_mode=geometry_mode,
        analysis_max_side=analysis_max_side,
        model_count=len(model_specs),
        include_model_confidence=include_model_confidence,
        include_model_output_confidence=include_model_output_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
    )

    def run_sequential():
        for index, record in enumerate(records, start=1):
            if cancel_check is not None and cancel_check():
                raise BuildCancelledError("Build cancelled")
            if state_callback is not None:
                state_callback(record.key, "running")
            payload = _analyze_record_payload(
                record,
                model_specs,
                analysis_max_side,
                geometry_mode,
                point_match_radius,
                boundary_radius,
                confidence_uncertainty_delta,
                point_confidence_radius,
                polygon_confidence_summary,
                cache_enabled,
                include_model_confidence,
                include_model_output_confidence,
                include_pairwise_metrics,
                comparison_pairs,
            )
            if state_callback is not None:
                state_callback(record.key, "done")
            if progress_callback is not None:
                progress_callback(index, len(records), record.key)
            yield record.key, payload

    if worker_count <= 1 or len(records) <= 1:
        yield from run_sequential()
        return

    try:
        executor_cls = ThreadPoolExecutor if use_thread_pool else ProcessPoolExecutor
        total_records = len(records)
        batch_size = 1 if use_thread_pool else _analytics_batch_size(total_records, worker_count)
        batch_count = int(math.ceil(float(total_records) / float(batch_size)))

        def make_work_item(record: FrameRecord):
            return (
                record,
                model_specs,
                analysis_max_side,
                geometry_mode,
                point_match_radius,
                boundary_radius,
                confidence_uncertainty_delta,
                point_confidence_radius,
                polygon_confidence_summary,
                cache_enabled,
                include_model_confidence,
                include_model_output_confidence,
                include_pairwise_metrics,
                comparison_pairs,
            )

        def make_batch(order_index: int):
            start = int(order_index) * int(batch_size)
            stop = min(total_records, start + int(batch_size))
            return tuple(make_work_item(records[index]) for index in range(start, stop))

        executor = executor_cls(max_workers=worker_count)
        shutdown_wait = True
        executor_shutdown = False
        try:
            completed = 0
            max_in_flight = max(1, worker_count)
            next_index = 0
            future_to_batch: dict[object, tuple[tuple[object, ...], ...]] = {}
            last_result_at = perf_counter()
            last_stall_progress_at = last_result_at

            def submit_work_item(order_index: int) -> None:
                batch = make_batch(order_index)
                if use_thread_pool:
                    future = executor.submit(_analyze_record_payload_for_executor, batch[0])
                else:
                    future = executor.submit(_analyze_record_payload_batch_for_executor, batch)
                future_to_batch[future] = batch
                if state_callback is not None:
                    for item in batch:
                        state_callback(item[0].key, "running")

            while next_index < batch_count and len(future_to_batch) < max_in_flight:
                submit_work_item(next_index)
                next_index += 1

            while future_to_batch:
                if cancel_check is not None and cancel_check():
                    if state_callback is not None:
                        for batch in future_to_batch.values():
                            for item in batch:
                                state_callback(item[0].key, "stale")
                    shutdown_wait = False
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor_shutdown = True
                    raise BuildCancelledError("Build cancelled")
                done, _pending = wait(
                    tuple(future_to_batch.keys()),
                    timeout=ANALYTICS_WAIT_TIMEOUT_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    now = perf_counter()
                    if (
                        progress_callback is not None
                        and now - last_stall_progress_at >= ANALYTICS_STALL_PROGRESS_SECONDS
                    ):
                        last_stall_progress_at = now
                        progress_callback(completed, len(records), "Waiting for analytics workers")
                    if now - last_result_at >= ANALYTICS_STALL_TIMEOUT_SECONDS:
                        if state_callback is not None:
                            for batch in future_to_batch.values():
                                for item in batch:
                                    state_callback(item[0].key, "stale")
                        shutdown_wait = False
                        executor.shutdown(wait=False, cancel_futures=True)
                        executor_shutdown = True
                        raise TimeoutError(
                            "Analytics workers stalled: no completed frame payloads within "
                            f"{ANALYTICS_STALL_TIMEOUT_SECONDS:.0f}s."
                        )
                    continue
                last_result_at = perf_counter()
                for future in done:
                    future_to_batch.pop(future)
                    result = future.result()
                    batch_result = (result,) if use_thread_pool else tuple(result)
                    if state_callback is not None:
                        for record_key, _payload in batch_result:
                            state_callback(record_key, "done")
                    for record_key, payload in batch_result:
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(completed, len(records), record_key)
                        yield record_key, payload
                    if completed % 32 == 0:
                        _clear_runtime_image_caches()
                while next_index < batch_count and len(future_to_batch) < max_in_flight:
                    submit_work_item(next_index)
                    next_index += 1
        except Exception:
            shutdown_wait = False
            if state_callback is not None:
                for batch in future_to_batch.values():
                    for item in batch:
                        state_callback(item[0].key, "stale")
            if not executor_shutdown:
                executor.shutdown(wait=False, cancel_futures=True)
                executor_shutdown = True
            raise
        finally:
            if not executor_shutdown:
                executor.shutdown(wait=shutdown_wait, cancel_futures=not shutdown_wait)
    except (PermissionError, OSError):
        yield from run_sequential()


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
    include_model_output_confidence: bool = False,
    include_pairwise_metrics: bool = True,
    comparison_pairs: tuple[ComparisonPairSelection, ...] = (),
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
        include_model_output_confidence=include_model_output_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
        comparison_pairs=comparison_pairs,
        progress_callback=progress_callback,
        state_callback=state_callback,
        cancel_check=cancel_check,
    ):
        if payload is not None:
            results[record_key] = payload
    return results


def _collect_frame_records_modern(
    model_specs: tuple[ModelSpec, ...],
    options: BuildOptions,
    *,
    original_folder: FolderSpec | None = None,
    cancel_check=None,
    progress_callback=None,
) -> BuildResult:
    if not model_specs:
        raise ValueError("At least one model folder must be selected.")

    extensions = tuple(str(ext).lower() for ext in options.file_extensions)
    base_spec = model_specs[0]
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError("Build cancelled")
    if progress_callback is not None:
        progress_callback(0, 0, "Indexing base folder")
    base_index = build_folder_index(
        base_spec.mask_folder,
        recursive=options.recursive,
        extensions=extensions,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        progress_interval=max(1, int(options.progress_update_interval)),
    )
    if not base_index:
        raise ValueError("Selected model folders do not contain matching image frames.")

    if progress_callback is not None:
        progress_callback(0, len(base_index), "Indexing auxiliary folders")
    original_index = (
        build_folder_index(
            original_folder.path,
            recursive=options.recursive,
            extensions=extensions,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            progress_interval=max(1, int(options.progress_update_interval)),
        )
        if original_folder is not None
        else {}
    )
    original_lookup = _build_folder_path_lookup(original_index)
    fallback_model_indexes: dict[str, _FolderPathLookup] = {}
    fallback_prob_indexes: dict[str, _FolderPathLookup] = {}

    records: list[FrameRecord] = []
    sorted_keys = sorted(base_index.keys(), key=natural_sort_key)
    for index, key in enumerate(sorted_keys):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        model_mask_paths: dict[str, str] = {base_spec.model_id: str(base_index[key])}
        model_prob_paths: dict[str, str] = {}
        if base_spec.prob_folder is not None:
            resolved_prob = _resolve_model_path_for_key(
                base_spec.prob_folder,
                key,
                extensions,
                fallback_prob_indexes,
                recursive=options.recursive,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            model_prob_paths[base_spec.model_id] = str(resolved_prob) if resolved_prob is not None else ""
        else:
            model_prob_paths[base_spec.model_id] = ""
        missing_required_model = False
        for spec in model_specs[1:]:
            resolved = _resolve_model_path_for_key(
                spec.mask_folder,
                key,
                extensions,
                fallback_model_indexes,
                recursive=options.recursive,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            if resolved is None:
                missing_required_model = True
                break
            model_mask_paths[spec.model_id] = str(resolved)
            if spec.prob_folder is not None:
                resolved_prob = _resolve_model_path_for_key(
                    spec.prob_folder,
                    key,
                    extensions,
                    fallback_prob_indexes,
                    recursive=options.recursive,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                )
                model_prob_paths[spec.model_id] = str(resolved_prob) if resolved_prob is not None else ""
            else:
                model_prob_paths[spec.model_id] = ""
        if missing_required_model:
            continue
        original_path = _resolve_aux_path_from_lookup(key, original_lookup)
        records.append(
            FrameRecord(
                key=key,
                display_name=Path(key).name,
                identity=build_frame_identity(key, index),
                first_path=str(base_index[key]),
                second_path=str(model_mask_paths.get(model_specs[1].model_id, model_mask_paths[base_spec.model_id]))
                if len(model_specs) > 1
                else str(base_index[key]),
                base_path=str(original_path) if original_path is not None else None,
                original_path=str(original_path) if original_path is not None else None,
                model_mask_paths=model_mask_paths,
                model_prob_paths=model_prob_paths,
            )
        )
        if progress_callback is not None and (
            index == 0
            or (index + 1) % max(1, int(options.progress_update_interval)) == 0
            or index + 1 == len(sorted_keys)
        ):
            progress_callback(index + 1, len(sorted_keys), key)
    return BuildResult(
        records=tuple(records),
        model_specs=model_specs,
        original_folder=original_folder,
        first_folder=FolderSpec(path=model_specs[0].mask_folder, label=model_specs[0].display_name)
        if len(model_specs) > 0
        else None,
        second_folder=FolderSpec(path=model_specs[1].mask_folder, label=model_specs[1].display_name)
        if len(model_specs) > 1
        else None,
        base_folder=original_folder,
        options=options,
        min_score=0.0,
        max_score=0.0,
        eligible_key_count=len(records),
        scores_computed=False,
        best_match_key=None,
        min_absolute_score=None,
        max_absolute_score=None,
        selected_metric_key="overall_frame_score",
        available_metric_keys=_available_metric_keys_for_models(model_specs, records),
    )


def collect_frame_records(
    model_specs: tuple[ModelSpec, ...] | FolderSpec,
    options: BuildOptions | FolderSpec,
    maybe_options: BuildOptions | None = None,
    *,
    original_folder: FolderSpec | None = None,
    base_folder: FolderSpec | None = None,
    cancel_check=None,
    progress_callback=None,
) -> BuildResult:
    """Collect matched frame records in modern or legacy-lite mode."""

    if isinstance(model_specs, FolderSpec):
        first_folder = model_specs
        second_folder = options if isinstance(options, FolderSpec) else None
        build_options = maybe_options if isinstance(maybe_options, BuildOptions) else BuildOptions()
        if second_folder is None:
            raise ValueError("Legacy collect_frame_records requires two folder specs.")
        compat_model_specs = (
            ModelSpec(
                model_id="first",
                display_name=first_folder.label,
                mask_folder=first_folder.path,
                threshold=float(build_options.mask_threshold),
            ),
            ModelSpec(
                model_id="second",
                display_name=second_folder.label,
                mask_folder=second_folder.path,
                threshold=float(build_options.mask_threshold),
            ),
        )
        modern = _collect_frame_records_modern(
            compat_model_specs,
            build_options,
            original_folder=base_folder,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        records: list[FrameRecord] = []
        for record in modern.records:
            first_path = str(record.model_mask_paths.get("first", ""))
            second_path = str(record.model_mask_paths.get("second", ""))
            records.append(
                replace(
                    record,
                    first_path=first_path,
                    second_path=second_path,
                    base_path=record.original_path,
                )
            )
        return replace(
            modern,
            records=tuple(records),
            first_folder=first_folder,
            second_folder=second_folder,
            base_folder=base_folder,
            original_folder=base_folder,
        )

    resolved_options = options if isinstance(options, BuildOptions) else maybe_options
    if not isinstance(resolved_options, BuildOptions):
        raise ValueError("collect_frame_records requires BuildOptions for modern mode.")
    return _collect_frame_records_modern(
        tuple(model_specs),
        resolved_options,
        original_folder=original_folder,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def compute_build_result_analytics(
    build_result: BuildResult,
    *,
    metric_key: str | None = None,
    excluded_record_keys: set[str] | None = None,
    progress_callback=None,
    state_callback=None,
    cancel_check=None,
) -> BuildResult:
    _clear_runtime_image_caches()
    records = list(build_result.records)
    if not records:
        return replace(build_result, scores_computed=True)
    if (
        len(build_result.model_specs) == 1
        and str(getattr(build_result.model_specs[0], "model_id", "") or "") == "base_layer"
    ):
        return replace(
            build_result,
            records=tuple(
                replace(
                    record,
                    score=0.0,
                    absolute_score=None,
                    relative_score=None,
                    score_percentile=None,
                    score_ready=False,
                    summary=None,
                )
                for record in records
            ),
            scores_computed=False,
            best_match_key=None,
            min_absolute_score=None,
            max_absolute_score=None,
            selected_metric_key="overall_frame_score",
            available_metric_keys=("overall_frame_score",),
        )

    active_metric = metric_key or build_result.selected_metric_key or "overall_frame_score"
    available_metric_keys_at_start = set(build_result.available_metric_keys or ()) | set(
        _available_metric_keys_for_models(build_result.model_specs, records)
    )
    if (
        (_metric_requires_model_confidence(active_metric) or _metric_requires_model_output_confidence(active_metric))
        and active_metric not in available_metric_keys_at_start
        and parse_confidence_pair_metric_key(active_metric) is None
        and parse_combined_pair_metric_key(active_metric) is None
    ):
        active_metric = "overall_frame_score"
    include_model_confidence = _metric_requires_model_confidence(active_metric)
    comparison_pairs = _normalized_comparison_pairs(tuple(getattr(build_result.options, "comparison_pairs", ()) or ()))
    comparison_target = getattr(build_result.options, "comparison_target", ComparisonTarget.OUTPUTS)
    comparison_target_value = str(
        getattr(comparison_target, "value", comparison_target) or ComparisonTarget.OUTPUTS.value
    )
    include_model_output_confidence = _metric_requires_model_output_confidence(
        active_metric
    ) or comparison_target_value in {ComparisonTarget.CONFIDENCE.value, ComparisonTarget.BOTH.value}
    include_pairwise_metrics = _metric_requires_pairwise_metrics(active_metric) or bool(comparison_pairs)
    excluded_keys = {str(key) for key in (excluded_record_keys or set()) if str(key)}
    records_to_analyze = [record for record in records if str(record.key) not in excluded_keys]
    records_by_key = {str(record.key): record for record in records}
    updated_records_by_key: dict[str, FrameRecord] = {}
    for record in records:
        record_key = str(record.key)
        if record_key not in excluded_keys:
            continue
        updated_records_by_key[record_key] = replace(
            record,
            score=0.0,
            absolute_score=None,
            relative_score=None,
            score_percentile=None,
            score_ready=False,
            summary=None,
        )
    try:
        payload_iter = _iter_record_payloads(
            records_to_analyze,
            build_result.model_specs,
            build_result.options.analysis_max_side,
            build_result.options.max_workers,
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
            cache_enabled=bool(build_result.options.cache_enabled),
            include_model_confidence=include_model_confidence,
            include_model_output_confidence=include_model_output_confidence,
            include_pairwise_metrics=include_pairwise_metrics,
            comparison_pairs=comparison_pairs,
            progress_callback=progress_callback,
            state_callback=state_callback,
            cancel_check=cancel_check,
        )
        for record_key, payload in payload_iter:
            if payload is None:
                continue
            record = records_by_key.get(str(record_key))
            if record is None:
                continue
            disagreement = float(payload.get("disagreement", 0.0))
            model_model_score = float(payload.get("model_model_score", 0.0))
            frame_type = "point" if str(payload.get("geometry_mode")) == GeometryMode.POINT.value else "polygon"
            model_diagnostics = payload.get("model_diagnostics") or {}
            model_confidence = payload.get("model_confidence") or {}
            model_confidence_output = payload.get("model_confidence_output") or {}
            confidence_pairwise_metrics = payload.get("confidence_pairwise_metrics") or {}
            confidence_pair_metric_values = payload.get("confidence_pair_metric_values") or {}
            combined_pair_metric_values = payload.get("combined_pair_metric_values") or {}
            pair_metric_values = payload.get("pair_metric_values") or {}
            comparison_result = payload.get("comparison_result")
            pairwise_rows = tuple(payload.get("pairwise_rows", ()))
            polygon_scores = _aggregate_inter_model_polygon_scores(pairwise_rows) if frame_type != "point" else {}
            point_scores = (
                _aggregate_inter_model_point_scores(pairwise_rows, float(build_result.options.point_match_radius))
                if frame_type == "point"
                else {}
            )

            export_priority = float(_clip01(disagreement))
            metric_values = {
                "overall_frame_score": export_priority,
                "export_priority_score": export_priority,
                "model_model_score": model_model_score,
                "disagreement_score": disagreement,
            }
            metric_values.update(polygon_scores)
            metric_values.update(point_scores)
            metric_values.update({str(key): float(value) for key, value in confidence_pairwise_metrics.items()})
            metric_values.update({str(key): float(value) for key, value in confidence_pair_metric_values.items()})
            metric_values.update({str(key): float(value) for key, value in combined_pair_metric_values.items()})
            metric_values.update({str(key): float(value) for key, value in pair_metric_values.items()})
            metric_values.update(
                _comparison_metric_values(
                    comparison_result if isinstance(comparison_result, FrameComparisonResult) else None
                )
            )
            for model_id, confidence_row in model_confidence.items():
                if hasattr(confidence_row, "mean_object_confidence"):
                    metric_values[_model_metric_key("model_confidence", str(model_id))] = float(
                        getattr(confidence_row, "frame_uncertainty_score", 0.0)
                    )
                    metric_values[_model_metric_key("model_uncertain_fraction", str(model_id))] = float(
                        getattr(confidence_row, "uncertain_fraction", 0.0)
                    )
                elif hasattr(confidence_row, "mean_point_confidence"):
                    metric_values[_model_metric_key("model_confidence", str(model_id))] = float(
                        getattr(confidence_row, "frame_uncertainty_score", 0.0)
                    )
                    metric_values[_model_metric_key("model_point_contrast", str(model_id))] = float(
                        getattr(confidence_row, "mean_point_contrast", 0.0)
                    )
            for model_id, confidence_output_row in model_confidence_output.items():
                metric_values[_model_metric_key("model_output_confidence", str(model_id))] = float(
                    getattr(confidence_output_row, "frame_uncertainty_score", 0.0)
                )

            summary = FrameAnalysisSummary(
                disagreement_score=float(disagreement),
                temporal_instability=0.0,
                structural_anomaly=0.0,
                export_priority_score=float(export_priority),
                metric_values=metric_values,
                model_confidence=model_confidence,
                model_confidence_output=model_confidence_output,
                model_diagnostics=model_diagnostics,
                pairwise_metrics=pairwise_rows,
                notes=(),
                frame_type=frame_type,
            )
            updated_records_by_key[str(record.key)] = replace(record, summary=summary)
    finally:
        _clear_runtime_image_caches()

    updated_records = [
        updated_records_by_key[str(record.key)] for record in records if str(record.key) in updated_records_by_key
    ]
    metric_is_required = _metric_requires_model_confidence(active_metric) or _metric_requires_model_output_confidence(
        active_metric
    )
    metric_values_by_key: dict[str, float] = {}
    for record in updated_records:
        value = _record_metric_value(record.summary, active_metric)
        if value is None and metric_is_required:
            continue
        metric_values_by_key[str(record.key)] = float(value or 0.0)
    absolute_scores = list(metric_values_by_key.values())
    min_absolute = min(absolute_scores) if absolute_scores else 0.0
    max_absolute = max(absolute_scores) if absolute_scores else 0.0
    span = max(EPS, max_absolute - min_absolute)
    higher_is_better = metric_higher_is_better(active_metric)

    scored_records: list[FrameRecord] = []
    best_key = None
    best_value = None
    for record in updated_records:
        absolute = metric_values_by_key.get(str(record.key))
        if absolute is None:
            scored_records.append(
                replace(
                    record,
                    score=0.0,
                    absolute_score=None,
                    relative_score=None,
                    score_percentile=None,
                    score_ready=False,
                )
            )
            continue
        relative = 0.0 if abs(max_absolute - min_absolute) <= EPS else (absolute - min_absolute) / span
        display = relative if higher_is_better else (1.0 - relative)
        scored = replace(
            record,
            score=float(display),
            absolute_score=float(absolute),
            relative_score=float(relative),
            score_ready=True,
        )
        scored_records.append(scored)
        if best_value is None or (absolute > best_value if higher_is_better else absolute < best_value):
            best_value = absolute
            best_key = record.key

    percentile_map = compute_metric_percentiles(scored_records, active_metric)
    scored_records = [
        replace(record, score_percentile=float(percentile_map.get(record.key, 0.0))) for record in scored_records
    ]
    available_metric_keys: list[str] = []
    seen_metric_keys: set[str] = set()
    for metric_key_candidate in _available_metric_keys_for_models(build_result.model_specs, scored_records):
        if metric_key_candidate not in seen_metric_keys:
            available_metric_keys.append(metric_key_candidate)
            seen_metric_keys.add(metric_key_candidate)
    for record in scored_records:
        summary = record.summary
        if summary is None:
            continue
        for metric_key_candidate in summary.metric_values.keys():
            metric_key_str = str(metric_key_candidate)
            if metric_key_str not in seen_metric_keys:
                available_metric_keys.append(metric_key_str)
                seen_metric_keys.add(metric_key_str)
    return replace(
        build_result,
        records=tuple(scored_records),
        min_score=min((record.score for record in scored_records), default=0.0),
        max_score=max((record.score for record in scored_records), default=0.0),
        scores_computed=True,
        best_match_key=best_key,
        min_absolute_score=min_absolute,
        max_absolute_score=max_absolute,
        selected_metric_key=active_metric,
        available_metric_keys=tuple(available_metric_keys),
    )


def load_frame_layers(record: FrameRecord) -> dict[str, object]:
    """Legacy-lite helper returning first/second/base grayscale and binary layers."""

    first_path = str(record.first_path or "")
    second_path = str(record.second_path or "")
    if not first_path and record.model_mask_paths:
        first_path = str(next(iter(record.model_mask_paths.values()), ""))
    if not second_path and len(record.model_mask_paths) > 1:
        second_path = str(list(record.model_mask_paths.values())[1])
    if not second_path:
        second_path = first_path
    first_gray = load_grayscale_image(Path(first_path))
    second_gray = load_grayscale_image(Path(second_path))
    if second_gray.shape != first_gray.shape:
        second_gray = resize_grayscale_image(second_gray, tuple(int(v) for v in first_gray.shape))
    base_gray = None
    base_path = record.base_path or record.original_path
    if base_path:
        base_gray = load_grayscale_image(Path(base_path))
        if base_gray.shape != first_gray.shape:
            base_gray = resize_grayscale_image(base_gray, tuple(int(v) for v in first_gray.shape))
    return {
        "first_gray": first_gray.copy(),
        "second_gray": second_gray.copy(),
        "first_binary": np.asarray(first_gray >= 128, dtype=bool),
        "second_binary": np.asarray(second_gray >= 128, dtype=bool),
        "base_gray": None if base_gray is None else base_gray.copy(),
        "shape": tuple(int(value) for value in first_gray.shape),
    }


def compute_build_result_metrics(
    build_result: BuildResult,
    *,
    comparison_mode: ComparisonMode | None = None,
    display_metric: str = "relative",
    progress_callback=None,
    cancel_check=None,
) -> BuildResult:
    """Legacy-lite score-computation wrapper implemented on top of analytics."""

    mode = comparison_mode or build_result.options.comparison_mode
    normalized_records: list[FrameRecord] = []
    total = max(1, len(build_result.records))
    for index, record in enumerate(build_result.records, start=1):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        layers = load_frame_layers(record)
        score = compute_comparison_score(layers["first_binary"], layers["second_binary"], mode)
        normalized_records.append(
            replace(
                record,
                absolute_score=float(score),
                score=float(score),
                score_ready=True,
            )
        )
        if progress_callback is not None:
            progress_callback(index, total, record.key)
    if not normalized_records:
        return replace(build_result, options=replace(build_result.options, comparison_mode=mode), scores_computed=True)
    absolute_scores = [float(record.absolute_score or 0.0) for record in normalized_records]
    min_absolute = float(min(absolute_scores))
    max_absolute = float(max(absolute_scores))
    span = max(EPS, max_absolute - min_absolute)
    use_absolute = str(display_metric or "relative").lower() == "absolute"
    updated_records = []
    for record in normalized_records:
        absolute = float(record.absolute_score or 0.0)
        relative = 0.0 if abs(max_absolute - min_absolute) <= EPS else float((absolute - min_absolute) / span)
        display = absolute if use_absolute else relative
        updated_records.append(replace(record, score=float(display), relative_score=float(relative), score_ready=True))
    best_record = min(updated_records, key=lambda item: float(item.absolute_score or 0.0))
    return replace(
        build_result,
        records=tuple(updated_records),
        options=replace(build_result.options, comparison_mode=mode),
        min_score=min((record.score for record in updated_records), default=0.0),
        max_score=max((record.score for record in updated_records), default=0.0),
        scores_computed=True,
        best_match_key=best_record.key,
        min_absolute_score=min_absolute,
        max_absolute_score=max_absolute,
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

    initial = collect_frame_records(
        first_folder, second_folder, options, base_folder=base_folder, cancel_check=cancel_check
    )
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
