"""_core: verify peak."""
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import replace
import numpy as np

from contour.application.use_cases.processing._core_parts.build_via_vectorization_mask import _ViaCandidate

from contour.application.use_cases.processing._core_parts.edge_ring_metrics import (
    _edge_ring_metrics,
    _isolation_score,
    _radial_contrast,
)

from contour.application.use_cases.processing._core_parts.extract_response_peaks import _component_extent

from contour.application.use_cases.processing._core_parts.refine_center import _refine_center

def _verify_peak(
    *,
    gray: np.ndarray,
    gradient: np.ndarray,
    blob_response: np.ndarray,
    ring_response: np.ndarray,
    cx: int,
    cy: int,
    radii: list[int],
    bright: bool,
    isolation: _IsolationContext,
) -> _ViaCandidate | None:
    height, width = gray.shape[:2]
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return None

    extent = _component_extent(isolation, cx, cy)
    if isolation.active and extent is None:
        # The peak sits outside every bright component - it cannot be a via
        # under the current polarity/range.
        return None

    best_score = -1.0
    best_candidate: _ViaCandidate | None = None
    for radius in radii:
        radius = max(2, int(radius))
        contrast = _radial_contrast(gray, cx, cy, radius, bright=bright)
        edge_strength, angular_coverage = _edge_ring_metrics(gradient, cx, cy, radius)

        # 80 units of absolute intensity difference is a confident via.
        contrast_score = max(0.0, min(1.0, contrast / 80.0))
        edge_score = max(0.0, min(1.0, edge_strength / 80.0))
        coverage_score = max(0.0, min(1.0, angular_coverage))

        isolation_score = _isolation_score(extent, radius)

        # Score from absolute circle-fit evidence only; peak response is used
        # for localization, not for scoring (otherwise any bright speckle on
        # texture wins because the response is percentile-normalized).
        score = contrast_score * 0.35 + edge_score * 0.25 + coverage_score * 0.20 + isolation_score * 0.20
        if score > best_score:
            best_score = score
            roundness = float(coverage_score * 100.0)
            best_candidate = _ViaCandidate(
                center_x=float(cx),
                center_y=float(cy),
                width=float(radius * 2 + 1),
                height=float(radius * 2 + 1),
                score=score,
                source="blob",
                roundness=roundness,
                contrast=float(contrast),
                edge_strength=float(edge_strength),
            )
    if best_candidate is not None:
        winning_radius = max(2, int((best_candidate.width - 1) // 2))
        refined_cx, refined_cy = _refine_center(gray, cx, cy, winning_radius, bright=bright)
        best_candidate = replace(best_candidate, center_x=refined_cx, center_y=refined_cy)
    return best_candidate

if TYPE_CHECKING:
    from contour.application.use_cases.processing._core_parts.extract_response_peaks import _IsolationContext
