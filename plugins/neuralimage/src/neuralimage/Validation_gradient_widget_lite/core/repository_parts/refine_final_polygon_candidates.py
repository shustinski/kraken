"""repository: refine final polygon candidates."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.candidate_border_span_features import (
    _candidate_axis_support_strength,
    _candidate_border_span_features,
    _candidate_boundary_separation,
    _candidate_has_compact_core,
    _local_prominence_map,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.carve_enclosed_low_probability import _carve_enclosed_low_probability_holes

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_erode

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _polygon_confidence_weak_threshold

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.retain_core_supported_region import _retain_core_supported_region

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.snap_elongated_candidate_boundaries import _snap_elongated_candidate_boundaries

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.split_candidate_by_spanning import _split_candidate_by_spanning_valleys

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.tighten_candidate_boundary_barriers import _tighten_candidate_with_boundary_barriers

def _refine_final_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    *,
    strong_mask: np.ndarray,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_transition_uncertainty_maps import _local_contrast_map
    if not candidates:
        return []
    low_threshold = _polygon_confidence_weak_threshold(float(strong_threshold), config)
    contrast_map = _local_contrast_map(probability, radius=1)
    prominence_map = _local_prominence_map(probability, radius=max(2, int(config.local_normalization_radius)))
    image_height, image_width = probability.shape
    refined_candidates: list[_PolygonConfidenceCandidate] = []
    for candidate in candidates:
        source_branch_set = set(candidate.source_branches)
        if 'large_polygon' in source_branch_set:
            # Large polygons may contain broad low-confidence texture that should remain
            # inside the polygon as uncertainty, not be carved into geometric holes.
            refined_mask = np.asarray(candidate.mask, dtype=bool)
        else:
            refined_mask = _carve_enclosed_low_probability_holes(
                probability,
                candidate.mask,
                low_threshold=low_threshold,
                config=config,
            )
            refined_mask = _snap_elongated_candidate_boundaries(
                probability,
                refined_mask,
                low_threshold=low_threshold,
                strong_threshold=strong_threshold,
                config=config,
                source_branches=candidate.source_branches,
            )
            refined_mask = _tighten_candidate_with_boundary_barriers(
                probability,
                refined_mask,
                strong_mask,
                low_threshold=low_threshold,
                strong_threshold=strong_threshold,
                config=config,
                source_branches=candidate.source_branches,
            )
        split_after_valleys = _split_candidate_by_spanning_valleys(
            probability,
            refined_mask,
            low_threshold=low_threshold,
            strong_threshold=strong_threshold,
            config=config,
            source_branches=candidate.source_branches,
        )
        for split_candidate in split_after_valleys:
            candidate_mask = np.asarray(split_candidate.mask, dtype=bool)
            candidate_after_holes = _make_polygon_candidate(probability, candidate_mask, split_candidate.source_branches)
            if candidate_after_holes is None:
                continue
            source_branch_set = set(split_candidate.source_branches)
            if 'large_polygon' in source_branch_set:
                # Large-polygon geometry is already reconstructed from a seed-connected local ROI
                # and optionally split by spanning valleys. Re-applying generic spill trimming here
                # collapses weak but valid wide polygons to bright core fragments.
                refined_candidates.append(candidate_after_holes)
                continue
            border_features = _candidate_border_span_features(
                candidate_mask,
                candidate_after_holes.bbox,
                probability.shape,
                cross_axis_max=float(config.spill_cross_axis_max),
            )
            touches_lr = bool(border_features['touches_lr'])
            touches_tb = bool(border_features['touches_tb'])
            touches_opposite_borders = touches_lr or touches_tb
            area_fraction = float(candidate_after_holes.area / max(1, probability.size))
            thin_cross_axis = bool(border_features['thin_cross_axis'])
            if touches_opposite_borders and (area_fraction >= float(config.spill_large_area_fraction) or thin_cross_axis) and float(candidate_after_holes.extent) >= float(config.spill_large_extent):
                interior_mask = _binary_erode(candidate_mask, 1) & candidate_mask
                texture_region = interior_mask if np.any(interior_mask) else candidate_mask
                texture_mean = float(np.mean(np.asarray(contrast_map[texture_region], dtype=np.float32), dtype=np.float64)) if np.any(texture_region) else 0.0
                peak_margin = float(max(0.0, float(candidate_after_holes.peak_probability) - float(candidate_after_holes.mean_probability)))
                boundary_separation = _candidate_boundary_separation(probability, candidate_mask)
                is_ribbon = float(max(candidate_after_holes.aspect_ratio, candidate_after_holes.elongation)) >= float(config.spill_ribbon_aspect_min)
                axis_coverage = float(border_features['axis_coverage'])
                object_prob = np.asarray(probability[candidate_mask], dtype=np.float32)
                local_quantile = float(np.quantile(object_prob, 0.80)) if object_prob.size else float(strong_threshold)
                core_threshold = max(
                    float(strong_threshold),
                    float(candidate_after_holes.mean_probability) + max(
                        float(config.spill_trim_delta),
                        0.35 * peak_margin,
                    ),
                    local_quantile,
                )
                strong_axis_coverage, strong_area_fraction = _candidate_axis_support_strength(
                    candidate_mask,
                    strong_mask,
                    touches_lr=touches_lr,
                    touches_tb=touches_tb,
                )
                fallback_only = source_branch_set == {'strong_mask'}
                preserve_compact_core = _candidate_has_compact_core(
                    probability,
                    candidate_mask,
                    core_threshold=core_threshold,
                    max_aspect_ratio=max(2.5, float(config.spill_ribbon_aspect_min)),
                )
                has_strong_axis_support = (
                    not fallback_only
                    and preserve_compact_core
                    and peak_margin > float(config.spill_peak_margin_max)
                    and (
                        strong_axis_coverage >= float(config.spill_strong_axis_coverage_min)
                        or strong_area_fraction >= float(config.spill_strong_area_fraction_min)
                    )
                )
                spill_like = (
                    is_ribbon
                    and axis_coverage >= float(config.spill_border_coverage_min)
                    and (peak_margin <= float(config.spill_peak_margin_max) or thin_cross_axis)
                    and (
                        texture_mean <= float(config.spill_low_texture_max)
                        or float(candidate_after_holes.mean_probability) <= float(config.spill_mean_probability_max)
                    )
                    and boundary_separation <= float(config.spill_boundary_separation_max)
                    and not has_strong_axis_support
                )
                if spill_like:
                    grow_threshold = max(
                        float(strong_threshold),
                        float(candidate_after_holes.mean_probability) + 0.5 * float(config.spill_trim_delta),
                    )
                    trimmed_mask = _retain_core_supported_region(
                        probability,
                        prominence_map,
                        candidate_mask,
                        core_threshold=core_threshold,
                        grow_threshold=grow_threshold,
                        prominence_threshold=float(config.spill_prominence_min),
                        max_core_aspect_ratio=max(2.5, float(config.spill_ribbon_aspect_min)),
                    )
                    trimmed_candidate = _make_polygon_candidate(probability, trimmed_mask, split_candidate.source_branches)
                    if trimmed_candidate is not None:
                        trimmed_border_features = _candidate_border_span_features(
                            trimmed_mask,
                            trimmed_candidate.bbox,
                            probability.shape,
                            cross_axis_max=float(config.spill_cross_axis_max),
                        )
                        trimmed_touches_opposite_borders = bool(trimmed_border_features['touches_lr']) or bool(trimmed_border_features['touches_tb'])
                        trimmed_axis_coverage = float(trimmed_border_features['axis_coverage'])
                        trimmed_peak_margin = float(max(0.0, float(trimmed_candidate.peak_probability) - float(trimmed_candidate.mean_probability)))
                        trimmed_boundary_separation = _candidate_boundary_separation(probability, trimmed_mask)
                        trimmed_is_ribbon = float(max(trimmed_candidate.aspect_ratio, trimmed_candidate.elongation)) >= float(config.spill_ribbon_aspect_min)
                        trimmed_spill_like = (
                            trimmed_is_ribbon
                            and trimmed_touches_opposite_borders
                            and trimmed_axis_coverage >= float(config.spill_border_coverage_min)
                            and trimmed_peak_margin <= float(config.spill_peak_margin_max)
                            and trimmed_boundary_separation <= float(config.spill_boundary_separation_max)
                        )
                        if trimmed_spill_like:
                            continue
                        candidate_after_holes = trimmed_candidate
                    else:
                        continue
            refined_candidates.append(candidate_after_holes)
    return refined_candidates

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _PolygonConfidenceCandidate
