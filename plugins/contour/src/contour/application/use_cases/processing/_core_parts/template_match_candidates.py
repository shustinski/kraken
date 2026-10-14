"""_core: template match candidates."""
from __future__ import annotations
import cv2
import numpy as np
from contour.application.processing import (
    ContourExtractionSettings,
)

from contour.application.use_cases.processing._core_parts.build_via_vectorization_mask import _ViaCandidate

from contour.application.use_cases.processing._core_parts.edge_ring_metrics import (
    _edge_ring_metrics,
    _normalize_template_for_matching,
    _radial_contrast,
)

from contour.application.use_cases.processing._core_parts.refine_center import _refine_center

def _template_match_candidates(
    *,
    gray: np.ndarray,
    gradient: np.ndarray,
    templates: list[np.ndarray],
    gate: np.ndarray,
    radii: list[int],
    expected_span: int,
    settings: ContourExtractionSettings,
    bright: bool,
) -> list[_ViaCandidate]:
    if not templates or gray.size == 0:
        return []
    search_image = _normalize_template_for_matching(gray)
    if cv2.countNonZero(gate) > 0:
        search_image = cv2.bitwise_and(search_image, gate)
    if cv2.countNonZero(search_image) == 0:
        return []
    min_score = max(0.0, min(1.0, float(settings.via_template_min_score)))
    candidates: list[_ViaCandidate] = []
    for template in templates:
        if search_image.shape[0] < template.shape[0] or search_image.shape[1] < template.shape[1]:
            continue
        prepared_template = _normalize_template_for_matching(template)
        scores = cv2.matchTemplate(search_image, prepared_template, cv2.TM_CCOEFF_NORMED)
        scores = np.nan_to_num(scores, copy=False)
        if scores.size == 0:
            continue
        cutoff = max(min_score, min(0.92, float(np.percentile(scores, 99.4))))
        local_max = scores >= (cv2.dilate(scores, np.ones((3, 3), dtype=np.uint8)) - 1e-6)
        ys, xs = np.where((scores >= cutoff) & local_max)
        if len(xs) > 600:
            values = scores[ys, xs]
            order = np.argsort(values)[-600:]
            xs = xs[order]
            ys = ys[order]
        radius = max(2, min(template.shape[:2]) // 3)
        for y_coord, x_coord in zip(ys, xs, strict=False):
            cx = int(x_coord) + template.shape[1] // 2
            cy = int(y_coord) + template.shape[0] // 2
            if cx >= gray.shape[1] or cy >= gray.shape[0]:
                continue
            if cv2.countNonZero(gate) > 0 and gate[cy, cx] == 0:
                continue
            refined_cx, refined_cy = _refine_center(gray, cx, cy, radius, bright=bright)
            refined_cx_int = round(refined_cx)
            refined_cy_int = round(refined_cy)
            contrast = _radial_contrast(gray, refined_cx_int, refined_cy_int, radius, bright=bright)
            edge_strength, coverage = _edge_ring_metrics(gradient, refined_cx_int, refined_cy_int, radius)
            template_score = float(scores[y_coord, x_coord])
            fit_score = (
                max(0.0, min(1.0, contrast / 80.0)) * 0.40
                + max(0.0, min(1.0, edge_strength / 80.0)) * 0.30
                + max(0.0, min(1.0, coverage)) * 0.30
            )
            score = max(fit_score, template_score * 0.6 + fit_score * 0.4)
            candidates.append(
                _ViaCandidate(
                    center_x=refined_cx,
                    center_y=refined_cy,
                    width=float(template.shape[1]),
                    height=float(template.shape[0]),
                    score=score,
                    source="template",
                    roundness=max(60.0, coverage * 100.0),
                    contrast=float(contrast),
                    edge_strength=float(edge_strength),
                )
            )
    return candidates
