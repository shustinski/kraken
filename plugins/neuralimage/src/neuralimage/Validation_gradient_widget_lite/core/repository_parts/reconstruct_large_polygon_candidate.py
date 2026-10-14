"""repository: reconstruct large polygon candidate."""
from __future__ import annotations
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.bridge_cut_mask import _connected_support_from_seed

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _binary_dilate_rect

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.spanning_barrier_mask import _spanning_barrier_mask

def _reconstruct_large_polygon_candidate(
    probability: np.ndarray,
    local_normalized_probability: np.ndarray,
    seed_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> _PolygonConfidenceCandidate | None:
    """Reconstruct one large polygon from a strong seed while keeping internal texture separate."""

    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_geometry import _binary_close_rect, _mask_bbox
    seed_bool = np.asarray(seed_mask, dtype=bool)
    if not np.any(seed_bool):
        return None
    seed_bbox = _mask_bbox(seed_bool)
    if seed_bbox[2] <= 0 or seed_bbox[3] <= 0:
        return None

    score_map = np.maximum(
        np.asarray(probability, dtype=np.float32),
        np.asarray(local_normalized_probability, dtype=np.float32),
    ).astype(np.float32)
    image_height, image_width = score_map.shape
    seed_x0, seed_y0, seed_w, seed_h = seed_bbox
    band_expand = max(
        int(config.large_polygon_band_expand),
        min(max(seed_w, seed_h), max(seed_w if seed_w < seed_h else seed_h, 1)),
    )
    roi_padding = max(1, int(config.large_polygon_roi_padding))
    base_support_threshold = max(
        float(config.hysteresis_low_floor),
        float(low_threshold) * float(config.large_polygon_low_scale),
    )
    seed_support_threshold = max(
        float(config.hysteresis_low_floor),
        min(
            float(base_support_threshold),
            float(low_threshold) * float(config.large_polygon_seed_low_scale),
            float(strong_threshold) * 0.40,
        ),
    )
    barrier_threshold = max(
        float(config.hysteresis_low_floor),
        float(base_support_threshold) - float(config.large_polygon_barrier_delta),
    )

    def _candidate_for_orientation(horizontal: bool) -> _PolygonConfidenceCandidate | None:
        envelope_support = np.asarray(score_map >= float(seed_support_threshold), dtype=bool)
        envelope_support |= seed_bool
        connected_envelope = _connected_support_from_seed(envelope_support, seed_bool)
        envelope_bbox = _mask_bbox(connected_envelope) if np.any(connected_envelope) else seed_bbox
        env_x0, env_y0, env_w, env_h = envelope_bbox
        env_x1 = env_x0 + env_w
        env_y1 = env_y0 + env_h
        if horizontal:
            roi_y0 = max(0, min(seed_y0, env_y0) - band_expand)
            roi_y1 = min(image_height, max(seed_y0 + seed_h, env_y1) + band_expand)
            roi_x0 = max(0, env_x0 - roi_padding)
            roi_x1 = min(image_width, env_x1 + roi_padding)
        else:
            roi_x0 = max(0, min(seed_x0, env_x0) - band_expand)
            roi_x1 = min(image_width, max(seed_x0 + seed_w, env_x1) + band_expand)
            roi_y0 = max(0, env_y0 - roi_padding)
            roi_y1 = min(image_height, env_y1 + roi_padding)
        if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
            return None
        score_roi = np.asarray(score_map[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
        seed_roi = np.asarray(seed_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
        if not np.any(seed_roi):
            return None

        support_roi = np.asarray(score_roi >= float(base_support_threshold), dtype=bool)
        if horizontal:
            support_roi = _binary_close_rect(
                support_roi,
                radius_y=max(0, int(config.large_polygon_minor_close_radius)),
                radius_x=max(1, int(config.large_polygon_major_close_radius)),
            )
        else:
            support_roi = _binary_close_rect(
                support_roi,
                radius_y=max(1, int(config.large_polygon_major_close_radius)),
                radius_x=max(0, int(config.large_polygon_minor_close_radius)),
            )
        support_roi |= seed_roi

        barrier_roi = _spanning_barrier_mask(
            score_roi,
            support_roi,
            threshold=barrier_threshold,
            horizontal=horizontal,
            coverage_min=float(config.large_polygon_barrier_coverage_min),
        )
        if np.any(barrier_roi):
            if horizontal:
                barrier_roi = _binary_dilate_rect(barrier_roi, radius_y=0, radius_x=1)
            else:
                barrier_roi = _binary_dilate_rect(barrier_roi, radius_y=1, radius_x=0)
            support_roi &= ~barrier_roi
            support_roi |= seed_roi

        connected_roi = _connected_support_from_seed(support_roi, seed_roi)
        if not np.any(connected_roi):
            return None
        candidate_mask = np.zeros_like(seed_bool, dtype=bool)
        candidate_mask[roi_y0:roi_y1, roi_x0:roi_x1] = connected_roi
        return _make_polygon_candidate(probability, candidate_mask, ('large_polygon',), roi_bbox=(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0))

    candidates = tuple(
        candidate
        for candidate in (
            _candidate_for_orientation(horizontal=True),
            _candidate_for_orientation(horizontal=False),
        )
        if candidate is not None
    )
    if not candidates:
        return None

    def _rank(candidate: _PolygonConfidenceCandidate) -> tuple[float, float, float, float]:
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        major_span = float(max(bbox_width, bbox_height))
        return (
            major_span,
            float(candidate.extent),
            float(max(candidate.aspect_ratio, candidate.elongation)),
            float(candidate.mean_probability),
        )

    return max(candidates, key=_rank)
