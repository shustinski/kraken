"""repository: internal confidence probability map."""
from __future__ import annotations
import hashlib
import pickle
from pathlib import Path
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    ANALYSIS_CACHE_DIR,
    DETAIL_CACHE_DIR,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
)
from neuralimage.Validation_gradient_widget_lite.core.confidence_maps import normalize_algorithmic_confidence
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildResult,
    FrameRecord,
    GeometryMode,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import (
    ANALYSIS_CACHE_MAX_FILES,
    DETAIL_CACHE_MAX_FILES,
    DETAIL_PAYLOAD_CACHE_SIZE,
    EPS,
    _DETAIL_PAYLOAD_MEMORY_CACHE,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _distance_transform

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.record_payload_cache_key import (
    _record_payload_cache_key,
    _record_payload_cache_path,
    _trim_pickle_cache_dir,
)

def _store_cached_record_payload(cache_key: str, payload: dict[str, object]) -> None:
    cache_path = _record_payload_cache_path(cache_key)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        _trim_pickle_cache_dir(ANALYSIS_CACHE_DIR, max_files=ANALYSIS_CACHE_MAX_FILES)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

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
    except Exception:
        return None
    _DETAIL_PAYLOAD_MEMORY_CACHE[cache_key] = payload
    _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
    while len(_DETAIL_PAYLOAD_MEMORY_CACHE) > DETAIL_PAYLOAD_CACHE_SIZE:
        _DETAIL_PAYLOAD_MEMORY_CACHE.popitem(last=False)
    return payload

def _store_cached_detail_payload(cache_key: str, payload: object) -> None:
    _DETAIL_PAYLOAD_MEMORY_CACHE[cache_key] = payload
    _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
    while len(_DETAIL_PAYLOAD_MEMORY_CACHE) > DETAIL_PAYLOAD_CACHE_SIZE:
        _DETAIL_PAYLOAD_MEMORY_CACHE.popitem(last=False)
    cache_path = _detail_payload_cache_path(cache_key)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        _trim_pickle_cache_dir(DETAIL_CACHE_DIR, max_files=DETAIL_CACHE_MAX_FILES)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

def _detail_payload_cache_key(record: FrameRecord, build_result: BuildResult, max_side: int | None, model_id: str | None) -> str:
    return _record_payload_cache_key(
        record,
        build_result.model_specs,
        max_side,
        build_result.options.geometry_mode,
        float(build_result.options.point_match_radius),
        int(getattr(build_result.options, 'boundary_radius', 1) or 1),
        float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
        int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS),
        str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED),
        include_model_confidence=True,
        include_pairwise_metrics=False,
        include_model_metrics=False,
    ) + f"::detail::{str(model_id or '')}"

def _detail_base_payload_cache_key(record: FrameRecord, build_result: BuildResult, max_side: int | None) -> str:
    return _record_payload_cache_key(
        record,
        build_result.model_specs,
        max_side,
        build_result.options.geometry_mode,
        float(build_result.options.point_match_radius),
        int(getattr(build_result.options, 'boundary_radius', 1) or 1),
        float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
        int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS),
        str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED),
        include_model_confidence=False,
        include_pairwise_metrics=False,
        include_model_metrics=False,
    ) + "::detail_base"

def _detail_confidence_cache_key(record: FrameRecord, build_result: BuildResult, max_side: int | None, model_id: str | None) -> str:
    return _detail_base_payload_cache_key(record, build_result, max_side) + f"::confidence::{str(model_id or '')}"

def _detail_confidence_payload_ready(confidence_row, geometry_mode: str) -> bool:
    if confidence_row is None:
        return False
    if geometry_mode == GeometryMode.POINT.value:
        return hasattr(confidence_row, 'mean_point_confidence')
    return hasattr(confidence_row, 'mean_object_confidence')

def _with_selected_detail_payload(payload: dict[str, object], target_model_id: str | None) -> dict[str, object]:
    detail = dict(payload)
    probabilities = detail.get('model_probabilities') or {}
    masks = detail.get('model_masks') or {}
    selected_model_id = target_model_id if target_model_id in probabilities else (next(iter(probabilities.keys()), None))
    fallback_prob = np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
    selected_prob = probabilities.get(selected_model_id, fallback_prob)
    selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
    detail['selected_model_id'] = selected_model_id
    detail['selected_prob'] = np.asarray(selected_prob, dtype=np.float32)
    detail['selected_mask'] = np.asarray(selected_mask, dtype=bool)
    return detail

def _mask_from_gray(gray: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    value = int(round(max(0.0, min(1.0, threshold)) * 255.0))
    return np.asarray(gray >= value, dtype=bool)

def _prob_from_gray(gray: np.ndarray) -> np.ndarray:
    return np.asarray(gray, dtype=np.float32) / 255.0

def _is_binary_like_probability(probability: np.ndarray, *, tolerance: float = 1e-4) -> bool:
    prob = np.asarray(probability, dtype=np.float32)
    if prob.size == 0:
        return False
    finite = prob[np.isfinite(prob)]
    if finite.size == 0:
        return False
    distance_to_binary = np.minimum(np.abs(finite), np.abs(finite - 1.0))
    return bool(np.max(distance_to_binary, initial=0.0) <= float(max(EPS, tolerance)))

def _internal_confidence_probability_map(
    probability: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
    allow_binary_proxy: bool = True,
) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    mask_bool = np.asarray(support_mask, dtype=bool) if support_mask is not None else np.asarray(prob >= 0.5, dtype=bool)
    if prob.size == 0 or not allow_binary_proxy or not _is_binary_like_probability(prob):
        return prob
    if mask_bool.shape != prob.shape:
        mask_bool = np.asarray(prob >= 0.5, dtype=bool)
    if not np.any(mask_bool):
        return prob
    inside = _distance_transform(mask_bool)
    outside = _distance_transform(~mask_bool)
    signed_distance = np.asarray(inside - outside, dtype=np.float32)
    scale = float(np.percentile(np.abs(signed_distance), 95.0))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.max(np.abs(signed_distance))) if signed_distance.size else 1.0
    scale = max(scale, 1.0)
    proxy = np.clip(0.5 + 0.5 * signed_distance / scale, 0.0, 1.0).astype(np.float32)
    proxy[mask_bool] = np.maximum(proxy[mask_bool], 0.5 + 0.5 * np.clip(inside[mask_bool] / scale, 0.0, 1.0))
    proxy[~mask_bool] = np.minimum(proxy[~mask_bool], 0.5 - 0.5 * np.clip(outside[~mask_bool] / scale, 0.0, 1.0))
    return np.clip(proxy, 0.0, 1.0).astype(np.float32)

def _confidence_map_from_probability(probability: np.ndarray) -> np.ndarray:
    return normalize_algorithmic_confidence(probability)
