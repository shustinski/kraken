"""repository: should reject branch spill."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.candidate_border_span_features import (
    _candidate_border_span_features,
    _candidate_boundary_separation,
    _candidate_has_compact_core,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_erode

def _should_reject_branch_spill(
    probability: np.ndarray,
    contrast_map: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    *,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> tuple[bool, tuple[str, ...]]:
    image_height, image_width = probability.shape
    border_features = _candidate_border_span_features(
        candidate.mask,
        candidate.bbox,
        probability.shape,
        cross_axis_max=float(config.spill_cross_axis_max),
    )
    touches_lr = bool(border_features['touches_lr'])
    touches_tb = bool(border_features['touches_tb'])
    if not (touches_lr or touches_tb):
        return False, ()
    area_fraction = float(candidate.area / max(1, probability.size))
    thin_cross_axis = bool(border_features['thin_cross_axis'])
    if area_fraction < float(config.spill_large_area_fraction) and not thin_cross_axis:
        return False, ()
    if float(candidate.extent) < float(config.spill_large_extent):
        return False, ()
    if float(max(candidate.aspect_ratio, candidate.elongation)) < float(config.spill_ribbon_aspect_min):
        return False, ()
    axis_coverage = float(border_features['axis_coverage'])
    if axis_coverage < float(config.spill_border_coverage_min):
        return False, ()
    mask_bool = np.asarray(candidate.mask, dtype=bool)
    boundary_separation = _candidate_boundary_separation(probability, mask_bool)
    if boundary_separation >= float(config.spill_boundary_separation_max):
        return False, ()
    interior_mask = _binary_erode(mask_bool, 1) & mask_bool
    texture_region = interior_mask if np.any(interior_mask) else mask_bool
    texture_mean = float(np.mean(np.asarray(contrast_map[texture_region], dtype=np.float32), dtype=np.float64)) if np.any(texture_region) else 0.0
    peak_margin = float(max(0.0, float(candidate.peak_probability) - float(candidate.mean_probability)))
    core_threshold = max(
        float(strong_threshold),
        float(candidate.mean_probability) + max(
            float(config.spill_trim_delta),
            0.35 * peak_margin,
        ),
    )
    has_compact_core = _candidate_has_compact_core(
        probability,
        mask_bool,
        core_threshold=core_threshold,
        max_aspect_ratio=max(1.6, float(config.spill_ribbon_aspect_min) * 0.55),
    )
    reject = (
        peak_margin <= float(config.spill_peak_margin_max)
        and (
            texture_mean <= float(config.spill_low_texture_max)
            or float(candidate.mean_probability) <= float(config.spill_mean_probability_max)
        )
        and not has_compact_core
    )
    if not reject:
        return False, ()
    return True, ('branch_spill_reject',)
