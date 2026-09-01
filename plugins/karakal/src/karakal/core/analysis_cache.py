"""Analysis and detail payload cache identities and storage."""

from __future__ import annotations

from .image_io import (
    _path_signature,
)

from .metric_keys import (
    _normalized_comparison_pairs,
)

from .repository_shared import (
    ANALYSIS_CACHE_DIR,
    ANALYSIS_CACHE_MAX_FILES,
    ANALYSIS_CACHE_VERSION,
    BuildResult,
    CACHE_TRIM_INTERVAL_SECONDS,
    ComparisonPairSelection,
    DETAIL_CACHE_DIR,
    DETAIL_CACHE_MAX_FILES,
    FrameRecord,
    GeometryMode,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    ModelSpec,
    OrderedDict,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    Path,
    _CACHE_TRIM_LAST_BY_DIR,
    _LOGGER,
    _active_performance_config,
    atomic_pickle_dump,
    estimate_size_bytes,
    hashlib,
    json,
    np,
    perf_counter,
    pickle,
    trim_directory_by_bytes,
)


def _record_payload_cache_key(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    *,
    include_model_confidence: bool,
    include_model_output_confidence: bool,
    include_pairwise_metrics: bool,
    comparison_pairs: tuple[ComparisonPairSelection, ...] = (),
) -> str:
    payload = {
        "version": ANALYSIS_CACHE_VERSION,
        "record_key": record.key,
        "analysis_max_side": int(analysis_max_side or 0),
        "geometry_mode": geometry_mode.value,
        "point_match_radius": float(point_match_radius),
        "boundary_radius": int(boundary_radius),
        "confidence_uncertainty_delta": float(confidence_uncertainty_delta),
        "point_confidence_radius": int(point_confidence_radius),
        "polygon_confidence_summary": str(polygon_confidence_summary),
        "include_model_confidence": bool(include_model_confidence),
        "include_model_output_confidence": bool(include_model_output_confidence),
        "include_pairwise_metrics": bool(include_pairwise_metrics),
        "comparison_pairs": [
            {
                "model_a_id": pair.model_a_id,
                "model_b_id": pair.model_b_id,
                "operations": list(pair.operations),
            }
            for pair in _normalized_comparison_pairs(comparison_pairs)
        ],
        "original": _path_signature(record.original_path),
        "models": [
            {
                "model_id": spec.model_id,
                "threshold": float(spec.threshold),
                "mask": _path_signature(record.model_mask_paths.get(spec.model_id)),
                "prob": _path_signature(record.model_prob_paths.get(spec.model_id)),
            }
            for spec in model_specs
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _record_payload_cache_path(cache_key: str) -> Path:
    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_CACHE_DIR / f"{cache_key}.pickle"


def _trim_pickle_cache_dir(cache_dir: Path, *, max_files: int, max_bytes: int | None = None) -> None:
    directory_key = str(Path(cache_dir))
    now = perf_counter()
    last_trim = _CACHE_TRIM_LAST_BY_DIR.get(directory_key, 0.0)
    if now - last_trim < CACHE_TRIM_INTERVAL_SECONDS:
        return
    _CACHE_TRIM_LAST_BY_DIR[directory_key] = now
    limit = max_bytes
    if limit is None:
        limit = int(_active_performance_config().disk_cache_limit_mb) * 1024 * 1024
    trim_directory_by_bytes(Path(cache_dir), max_bytes=limit, max_files=max_files)


def _load_cached_record_payload(cache_key: str) -> dict[str, object] | None:
    cache_path = _record_payload_cache_path(cache_key)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, TypeError, ValueError) as error:
        _LOGGER.warning("Ignoring corrupt analysis cache entry %s: %s", cache_path, error)
        return None
    return payload if isinstance(payload, dict) else None


def _store_cached_record_payload(cache_key: str, payload: dict[str, object]) -> None:
    cache_path = _record_payload_cache_path(cache_key)
    try:
        atomic_pickle_dump(cache_path, payload)
        _trim_pickle_cache_dir(ANALYSIS_CACHE_DIR, max_files=ANALYSIS_CACHE_MAX_FILES)
    except (OSError, pickle.PickleError, TypeError, ValueError) as error:
        _LOGGER.warning("Could not store analysis cache entry %s: %s", cache_path, error)


DETAIL_PAYLOAD_CACHE_SIZE = 32

_DETAIL_PAYLOAD_MEMORY_CACHE: OrderedDict[str, object] = OrderedDict()


def _trim_detail_memory_cache() -> None:
    limit_bytes = int(_active_performance_config().preview_cache_limit_mb) * 1024 * 1024
    total_bytes = sum(estimate_size_bytes(item) for item in _DETAIL_PAYLOAD_MEMORY_CACHE.values())
    while _DETAIL_PAYLOAD_MEMORY_CACHE and (
        len(_DETAIL_PAYLOAD_MEMORY_CACHE) > DETAIL_PAYLOAD_CACHE_SIZE or total_bytes > limit_bytes
    ):
        _key, evicted = _DETAIL_PAYLOAD_MEMORY_CACHE.popitem(last=False)
        total_bytes = max(0, total_bytes - estimate_size_bytes(evicted))


def _detail_payload_cache_path(cache_key: str) -> Path:
    DETAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
    return DETAIL_CACHE_DIR / f"{digest}.pickle"


def _load_cached_detail_payload(cache_key: str) -> object | None:
    payload = _DETAIL_PAYLOAD_MEMORY_CACHE.get(cache_key)
    if payload is not None:
        _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
        return payload
    cache_path = _detail_payload_cache_path(cache_key)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, TypeError, ValueError) as error:
        _LOGGER.warning("Ignoring corrupt detail cache entry %s: %s", cache_path, error)
        return None
    _DETAIL_PAYLOAD_MEMORY_CACHE[cache_key] = payload
    _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
    _trim_detail_memory_cache()
    return payload


def _store_cached_detail_payload(cache_key: str, payload: object) -> None:
    _DETAIL_PAYLOAD_MEMORY_CACHE[cache_key] = payload
    _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
    _trim_detail_memory_cache()
    cache_path = _detail_payload_cache_path(cache_key)
    try:
        atomic_pickle_dump(cache_path, payload)
        _trim_pickle_cache_dir(DETAIL_CACHE_DIR, max_files=DETAIL_CACHE_MAX_FILES)
    except (OSError, pickle.PickleError, TypeError, ValueError) as error:
        _LOGGER.warning("Could not store detail cache entry %s: %s", cache_path, error)


def _detail_payload_cache_key(
    record: FrameRecord, build_result: BuildResult, max_side: int | None, model_id: str | None
) -> str:
    return (
        _record_payload_cache_key(
            record,
            build_result.model_specs,
            max_side,
            build_result.options.geometry_mode,
            float(build_result.options.point_match_radius),
            int(getattr(build_result.options, "boundary_radius", 1) or 1),
            float(getattr(build_result.options, "confidence_uncertainty_delta", MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
            int(
                getattr(build_result.options, "point_confidence_radius", POINT_CONFIDENCE_NEIGHBOR_RADIUS)
                or POINT_CONFIDENCE_NEIGHBOR_RADIUS
            ),
            str(
                getattr(build_result.options, "polygon_confidence_summary", POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)
                or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
            ),
            include_model_confidence=True,
            include_model_output_confidence=False,
            include_pairwise_metrics=False,
        )
        + f"::detail::{str(model_id or '')}"
    )


def _detail_base_payload_cache_key(record: FrameRecord, build_result: BuildResult, max_side: int | None) -> str:
    return (
        _record_payload_cache_key(
            record,
            build_result.model_specs,
            max_side,
            build_result.options.geometry_mode,
            float(build_result.options.point_match_radius),
            int(getattr(build_result.options, "boundary_radius", 1) or 1),
            float(getattr(build_result.options, "confidence_uncertainty_delta", MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
            int(
                getattr(build_result.options, "point_confidence_radius", POINT_CONFIDENCE_NEIGHBOR_RADIUS)
                or POINT_CONFIDENCE_NEIGHBOR_RADIUS
            ),
            str(
                getattr(build_result.options, "polygon_confidence_summary", POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)
                or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
            ),
            include_model_confidence=False,
            include_model_output_confidence=False,
            include_pairwise_metrics=False,
        )
        + "::detail_base"
    )


def _detail_confidence_cache_key(
    record: FrameRecord, build_result: BuildResult, max_side: int | None, model_id: str | None
) -> str:
    return _detail_base_payload_cache_key(record, build_result, max_side) + f"::confidence::{str(model_id or '')}"


def _detail_confidence_payload_ready(confidence_row, geometry_mode: str) -> bool:
    if confidence_row is None:
        return False
    if geometry_mode == GeometryMode.POINT.value:
        return hasattr(confidence_row, "mean_point_confidence")
    return hasattr(confidence_row, "mean_object_confidence")


def _with_selected_detail_payload(payload: dict[str, object], target_model_id: str | None) -> dict[str, object]:
    detail = dict(payload)
    probabilities = detail.get("model_probabilities") or {}
    masks = detail.get("model_masks") or {}
    selected_model_id = (
        target_model_id if target_model_id in probabilities else (next(iter(probabilities.keys()), None))
    )
    fallback_prob = (
        np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
    )
    selected_prob = probabilities.get(selected_model_id, fallback_prob)
    selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
    detail["selected_model_id"] = selected_model_id
    detail["selected_prob"] = np.asarray(selected_prob, dtype=np.float32)
    detail["selected_mask"] = np.asarray(selected_mask, dtype=bool)
    return detail
