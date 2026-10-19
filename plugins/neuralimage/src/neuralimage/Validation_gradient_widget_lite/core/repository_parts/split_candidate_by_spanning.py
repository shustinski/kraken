"""repository: split candidate by spanning."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    PolygonConfidencePipelineConfig,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate,
    _binary_dilate_rect,
    _label_components,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.make_polygon_candidate import _make_polygon_candidate

def _split_candidate_by_spanning_valleys(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> tuple[_PolygonConfidenceCandidate, ...]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return ()
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return ()
    x0, y0, width, height = candidate.bbox
    if width < 6 and height < 6:
        return (candidate,)
    roi_mask = np.asarray(mask_bool[y0:y0 + height, x0:x0 + width], dtype=bool)
    roi_prob = np.asarray(probability[y0:y0 + height, x0:x0 + width], dtype=np.float32)
    horizontal = width >= height
    elongated_like = float(max(candidate.aspect_ratio, candidate.elongation)) >= float(config.boundary_snap_min_aspect)
    peak_margin = float(max(0.0, float(candidate.peak_probability) - float(candidate.mean_probability)))
    valley_threshold = max(
        float(config.hysteresis_low_floor),
        min(
            float(strong_threshold) - 0.08,
            float(low_threshold) + max(0.05, float(config.spill_trim_delta) + 0.02),
        ),
    )
    if 'large_polygon' in set(source_branches):
        valley_threshold = max(
            float(config.hysteresis_low_floor),
            min(
                float(strong_threshold) - 0.08,
                float(candidate.mean_probability) - max(0.04, 0.50 * peak_margin),
            ),
        )
    valley_mask = roi_mask & (roi_prob <= valley_threshold)
    if not np.any(valley_mask):
        return (candidate,)
    valley_labels, valley_count = _label_components(valley_mask)
    if valley_count <= 0:
        return (candidate,)
    carved_roi = roi_mask.copy()
    coverage_min = float(config.spill_border_coverage_min)
    minor_coverage_min = float(config.valley_minor_coverage_min)
    removed_any = False
    for label_id in range(1, int(valley_count) + 1):
        component = valley_labels == label_id
        if not np.any(component):
            continue
        ys, xs = np.nonzero(component)
        if ys.size == 0 or xs.size == 0:
            continue
        comp_width = int(np.max(xs) - np.min(xs) + 1)
        comp_height = int(np.max(ys) - np.min(ys) + 1)
        cover_x = float(comp_width / max(1, width))
        cover_y = float(comp_height / max(1, height))
        is_horizontal_separator = cover_x >= coverage_min and cover_y <= 0.35
        is_vertical_separator = cover_y >= coverage_min and cover_x <= 0.35
        if elongated_like and horizontal:
            is_vertical_separator = is_vertical_separator or (cover_y >= minor_coverage_min and cover_x <= 0.25)
        elif elongated_like and not horizontal:
            is_horizontal_separator = is_horizontal_separator or (cover_x >= minor_coverage_min and cover_y <= 0.25)
        if not (is_horizontal_separator or is_vertical_separator):
            continue
        # Slight dilation makes the separator robust to weak single-pixel bridges.
        if horizontal and is_vertical_separator:
            carved_roi &= ~_binary_dilate_rect(component, 1, 0)
        elif (not horizontal) and is_horizontal_separator:
            carved_roi &= ~_binary_dilate_rect(component, 0, 1)
        else:
            carved_roi &= ~_binary_dilate(component, 1)
        removed_any = True
    if not removed_any or not np.any(carved_roi):
        return (candidate,)
    split_labels, split_count = _label_components(carved_roi)
    if split_count <= 1:
        carved_mask = mask_bool.copy()
        carved_mask[y0:y0 + height, x0:x0 + width] = carved_roi
        split_candidate = _make_polygon_candidate(probability, carved_mask, source_branches)
        return (split_candidate,) if split_candidate is not None else (candidate,)
    split_candidates: list[_PolygonConfidenceCandidate] = []
    strong_mask = np.asarray(probability, dtype=np.float32) >= float(strong_threshold)
    for label_id in range(1, int(split_count) + 1):
        component_roi = split_labels == label_id
        if not np.any(component_roi):
            continue
        component_mask = np.zeros_like(mask_bool, dtype=bool)
        component_mask[y0:y0 + height, x0:x0 + width] = component_roi
        if not np.any(component_mask & strong_mask):
            continue
        component_candidate = _make_polygon_candidate(probability, component_mask, source_branches)
        if component_candidate is not None:
            split_candidates.append(component_candidate)
    return tuple(split_candidates) if split_candidates else (candidate,)

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _PolygonConfidenceCandidate
