"""repository: polygon confidence pipeline."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidenceDebugCandidate,
    PolygonConfidenceDebugData,
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.complete_polygon_candidates import _complete_polygon_candidates

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.extract_branch_candidates import (
    _candidate_object_labels,
    _extract_branch_candidates,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.extract_large_polygon_candidates import _extract_large_polygon_candidates

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_dilate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.merge_polygon_candidates import _merge_polygon_candidates

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import (
    _polygon_confidence_config,
    _polygon_confidence_weak_threshold,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.refine_final_polygon_candidates import _refine_final_polygon_candidates

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.should_reject_branch_spill import _should_reject_branch_spill

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.split_polygon_candidate_by import _split_polygon_candidate_by_barriers

def _polygon_confidence_pipeline(
    probability: np.ndarray,
    strong_mask: np.ndarray,
    *,
    strong_threshold: float = 0.5,
    config: PolygonConfidencePipelineConfig | None = None,
    include_debug: bool = False,
) -> tuple[np.ndarray, tuple[_PolygonConfidenceCandidate, ...], PolygonConfidenceDebugData | None]:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_geometry import _binary_close_rect, _normalize_probability_map, _preprocess_polygon_probability
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_transition_uncertainty_maps import _local_contrast_map, _local_mean_map
    cfg = config or _polygon_confidence_config()
    stage_started = perf_counter()
    stage_timings_ms: dict[str, float] = {}

    raw_prob = _normalize_probability_map(probability)
    preprocessed_prob, local_normalized_prob = _preprocess_polygon_probability(raw_prob, cfg)
    high_threshold = float(max(0.0, min(1.0, strong_threshold)))
    low_threshold = _polygon_confidence_weak_threshold(high_threshold, cfg)
    stage_timings_ms['preprocess'] = 1000.0 * (perf_counter() - stage_started)

    branch_started = perf_counter()
    global_low_mask = np.asarray(preprocessed_prob >= low_threshold, dtype=bool)
    global_high_mask = np.asarray(preprocessed_prob >= high_threshold, dtype=bool) | np.asarray(strong_mask, dtype=bool)
    branch_contrast_map = _local_contrast_map(preprocessed_prob, radius=1)

    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    next_debug_id = 1

    dominant_high_threshold = max(high_threshold, float(cfg.dominant_min_mean_probability))
    dominant_high_mask = np.asarray(preprocessed_prob >= dominant_high_threshold, dtype=bool) | np.asarray(strong_mask, dtype=bool)

    large_polygon_candidates, large_polygon_mask, large_polygon_debug, next_debug_id = _extract_large_polygon_candidates(
        preprocessed_prob,
        local_normalized_prob,
        dominant_high_mask,
        low_threshold=low_threshold,
        strong_threshold=high_threshold,
        config=cfg,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(large_polygon_debug)
    dominant_candidates = list(large_polygon_candidates)
    dominant_mask = np.asarray(large_polygon_mask, dtype=bool)
    dominant_lock_mask = (
        _binary_dilate(dominant_mask, max(0, int(cfg.dominant_lock_radius)))
        if np.any(dominant_mask)
        else np.zeros_like(global_low_mask, dtype=bool)
    )

    def _global_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        return (False, reject_notes) if reject else (True, ())

    global_candidates, global_mask, global_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        global_low_mask & ~dominant_lock_mask,
        global_high_mask & ~dominant_lock_mask,
        branch='global_hysteresis',
        min_area=1,
        require_high_core=True,
        acceptance_fn=_global_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(global_debug)

    elongated_low_mask = _binary_close_rect(global_low_mask, int(cfg.elongated_vertical_radius), int(cfg.elongated_horizontal_radius))
    elongated_low_mask |= _binary_close_rect(global_low_mask, int(cfg.elongated_horizontal_radius), int(cfg.elongated_vertical_radius))

    def _elongated_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        if candidate.area < max(1, int(cfg.elongated_min_area)):
            notes.append('elongated_area_below_min')
            return False, tuple(notes)
        if max(candidate.aspect_ratio, candidate.elongation) < float(cfg.elongated_min_aspect_ratio):
            notes.append('elongated_ratio_too_small')
            return False, tuple(notes)
        if not has_high_core:
            notes.append('missing_high_core')
            return False, tuple(notes)
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        if reject:
            notes.extend(reject_notes)
            return False, tuple(notes)
        return True, tuple(notes)

    elongated_candidates, elongated_mask, elongated_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        elongated_low_mask & ~dominant_lock_mask,
        global_high_mask & ~dominant_lock_mask,
        branch='elongated',
        min_area=int(cfg.elongated_min_area),
        require_high_core=True,
        acceptance_fn=_elongated_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(elongated_debug)

    small_low_threshold = max(float(cfg.small_mean_floor), low_threshold * float(cfg.small_low_scale))
    small_high_threshold = max(float(cfg.small_peak_floor), high_threshold * float(cfg.small_high_scale))
    small_low_mask = np.asarray(preprocessed_prob >= small_low_threshold, dtype=bool)
    small_high_mask = np.asarray(preprocessed_prob >= small_high_threshold, dtype=bool)

    def _small_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        peak_ok = candidate.peak_probability >= max(float(cfg.small_peak_floor), small_high_threshold)
        mean_ok = candidate.mean_probability >= float(cfg.small_mean_floor)
        if not (has_high_core or peak_ok):
            notes.append('small_missing_peak')
            return False, tuple(notes)
        if not mean_ok:
            notes.append('small_mean_too_low')
            return False, tuple(notes)
        return True, tuple(notes)

    small_candidates, small_mask, small_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        small_low_mask & ~dominant_lock_mask,
        small_high_mask & ~dominant_lock_mask,
        branch='small_weak',
        min_area=max(1, int(cfg.small_min_area)),
        max_area=max(1, int(cfg.small_max_area)),
        require_high_core=False,
        acceptance_fn=_small_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(small_debug)

    adaptive_background = _local_mean_map(preprocessed_prob, radius=max(1, int(cfg.adaptive_radius)))
    adaptive_response = np.asarray(preprocessed_prob - adaptive_background, dtype=np.float32)
    adaptive_low_mask = np.asarray(adaptive_response >= float(cfg.adaptive_low_offset), dtype=bool)
    adaptive_high_mask = np.asarray(adaptive_response >= float(cfg.adaptive_high_offset), dtype=bool)

    def _adaptive_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        if not has_high_core:
            notes.append('adaptive_missing_high_core')
            return False, tuple(notes)
        if candidate.mean_probability < max(float(cfg.small_mean_floor) * 0.9, 0.12) and candidate.peak_probability < max(float(cfg.small_peak_floor), 0.18):
            notes.append('adaptive_signal_too_low')
            return False, tuple(notes)
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        if reject:
            notes.extend(reject_notes)
            return False, tuple(notes)
        return True, tuple(notes)

    adaptive_candidates, adaptive_mask, adaptive_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        adaptive_low_mask & ~dominant_lock_mask,
        adaptive_high_mask & ~dominant_lock_mask,
        branch='adaptive_local',
        min_area=max(1, int(cfg.small_min_area)),
        require_high_core=True,
        acceptance_fn=_adaptive_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(adaptive_debug)

    candidates = list(large_polygon_candidates) + list(global_candidates) + list(elongated_candidates) + list(small_candidates) + list(adaptive_candidates)
    if not candidates and np.any(np.asarray(strong_mask, dtype=bool)):
        fallback_candidate = _make_polygon_candidate(raw_prob, np.asarray(strong_mask, dtype=bool), ('strong_mask',))
        if fallback_candidate is not None:
            candidates = [fallback_candidate]
    stage_timings_ms['branch_extract'] = 1000.0 * (perf_counter() - branch_started)

    merge_started = perf_counter()
    merged_candidates = _merge_polygon_candidates(raw_prob, candidates, cfg)
    stage_timings_ms['initial_merge'] = 1000.0 * (perf_counter() - merge_started)

    completion_started = perf_counter()
    seed_mask = global_high_mask | small_high_mask | adaptive_high_mask | np.asarray(strong_mask, dtype=bool)
    completed_candidates = _complete_polygon_candidates(
        raw_prob,
        merged_candidates,
        strong_threshold=high_threshold,
        weak_threshold=low_threshold,
        high_seed_mask=seed_mask,
    )
    stage_timings_ms['completion'] = 1000.0 * (perf_counter() - completion_started)

    split_started = perf_counter()
    split_candidates: list[_PolygonConfidenceCandidate] = []
    separation_boundary_cues = np.zeros_like(preprocessed_prob, dtype=np.float32)
    separation_core_mask = np.zeros_like(global_low_mask, dtype=bool)
    separation_candidate_region = np.zeros_like(global_low_mask, dtype=bool)
    separation_thin_barrier = np.zeros_like(global_low_mask, dtype=bool)
    separation_bridge_cuts = np.zeros_like(global_low_mask, dtype=bool)
    separation_barrier_stops = np.zeros_like(global_low_mask, dtype=bool)
    for candidate in completed_candidates:
        if 'large_polygon' in set(candidate.source_branches):
            split_candidates.append(candidate)
            continue
        separated, separation_debug = _split_polygon_candidate_by_barriers(
            raw_prob,
            candidate,
            seed_mask,
            low_threshold=low_threshold,
            strong_threshold=high_threshold,
            config=cfg,
            include_debug=include_debug,
        )
        split_candidates.extend(separated)
        if include_debug and separation_debug is not None:
            separation_boundary_cues = np.maximum(
                separation_boundary_cues,
                np.asarray(separation_debug.get("boundary_cues"), dtype=np.float32),
            )
            separation_core_mask |= np.asarray(separation_debug.get("core_seeds"), dtype=bool)
            separation_candidate_region |= np.asarray(separation_debug.get("candidate_region"), dtype=bool)
            separation_thin_barrier |= np.asarray(separation_debug.get("thin_barrier"), dtype=bool)
            separation_bridge_cuts |= np.asarray(separation_debug.get("bridge_cuts"), dtype=bool)
            separation_barrier_stops |= np.asarray(separation_debug.get("barrier_blocked"), dtype=bool)
    stage_timings_ms['split_refine'] = 1000.0 * (perf_counter() - split_started)

    final_merge_started = perf_counter()
    final_candidates = _merge_polygon_candidates(raw_prob, split_candidates or completed_candidates, cfg)
    final_candidates = _refine_final_polygon_candidates(
        raw_prob,
        final_candidates,
        strong_mask=np.asarray(strong_mask, dtype=bool),
        strong_threshold=high_threshold,
        config=cfg,
    )
    stage_timings_ms['final_merge'] = 1000.0 * (perf_counter() - final_merge_started)

    final_mask = np.zeros_like(np.asarray(strong_mask, dtype=bool), dtype=bool)
    for candidate in final_candidates:
        final_mask |= np.asarray(candidate.mask, dtype=bool)

    if not np.any(final_mask) and np.any(np.asarray(strong_mask, dtype=bool)) and not candidates:
        final_mask = np.asarray(strong_mask, dtype=bool).copy()
        fallback_candidate = _make_polygon_candidate(raw_prob, final_mask, ('strong_mask',))
        final_candidates = [fallback_candidate] if fallback_candidate is not None else []

    object_labels = _candidate_object_labels(final_candidates, final_mask.shape) if final_candidates else np.zeros(final_mask.shape, dtype=np.int32)

    debug_data = None
    if include_debug:
        final_debug_rows = list(debug_rows)
        for object_index, candidate in enumerate(final_candidates, start=1):
            final_debug_rows.append(PolygonConfidenceDebugCandidate(
                object_id=int(object_index),
                branch='merged',
                source_branches=tuple(candidate.source_branches),
                accepted=True,
                area=int(candidate.area),
                bbox_x=int(candidate.bbox[0]),
                bbox_y=int(candidate.bbox[1]),
                bbox_width=int(candidate.bbox[2]),
                bbox_height=int(candidate.bbox[3]),
                aspect_ratio=float(candidate.aspect_ratio),
                elongation=float(candidate.elongation),
                peak_probability=float(candidate.peak_probability),
                mean_probability=float(candidate.mean_probability),
                extent=float(candidate.extent),
                notes=('final_object',),
            ))
        debug_data = PolygonConfidenceDebugData(
            preprocessed_probability=np.asarray(preprocessed_prob, dtype=np.float32),
            locally_normalized_probability=np.asarray(local_normalized_prob, dtype=np.float32),
            boundary_cues=np.asarray(separation_boundary_cues, dtype=np.float32),
            low_mask=np.asarray(global_low_mask, dtype=bool),
            high_mask=np.asarray(global_high_mask, dtype=bool),
            adaptive_low_mask=np.asarray(adaptive_low_mask, dtype=bool),
            adaptive_high_mask=np.asarray(adaptive_high_mask, dtype=bool),
            core_seed_mask=np.asarray(separation_core_mask, dtype=bool),
            candidate_region_mask=np.asarray(separation_candidate_region, dtype=bool),
            thin_barrier_map=np.asarray(separation_thin_barrier, dtype=bool),
            bridge_cut_mask=np.asarray(separation_bridge_cuts, dtype=bool),
            barrier_stop_mask=np.asarray(separation_barrier_stops, dtype=bool),
            branch_masks={
                'large_polygon': np.asarray(large_polygon_mask, dtype=bool),
                'dominant_clear': np.asarray(dominant_mask, dtype=bool),
                'global_hysteresis': np.asarray(global_mask, dtype=bool),
                'elongated': np.asarray(elongated_mask, dtype=bool),
                'small_weak': np.asarray(small_mask, dtype=bool),
                'adaptive_local': np.asarray(adaptive_mask, dtype=bool),
                'core_seeds': np.asarray(separation_core_mask, dtype=bool),
                'candidate_region': np.asarray(separation_candidate_region, dtype=bool),
                'thin_barrier': np.asarray(separation_thin_barrier, dtype=bool),
                'bridge_cuts': np.asarray(separation_bridge_cuts, dtype=bool),
                'barrier_stops': np.asarray(separation_barrier_stops, dtype=bool),
            },
            merged_mask=np.asarray(final_mask, dtype=bool),
            object_labels=np.asarray(object_labels, dtype=np.int32),
            candidate_rows=tuple(final_debug_rows),
            timings_ms={name: float(value) for name, value in stage_timings_ms.items()},
        )

    return np.asarray(final_mask, dtype=bool), tuple(final_candidates), debug_data
