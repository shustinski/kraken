"""_core: apply via filters."""
from __future__ import annotations
import cv2
import numpy as np
from contour.utils import ensure_binary_mask
from contour.application.processing import (
    ContourExtractionSettings,
)

from contour.application.use_cases.processing._core_parts.build_via_vectorization_mask import _ViaCandidate

def _apply_via_filters(
    candidate: _ViaCandidate, settings: ContourExtractionSettings, expected_span: int
) -> _ViaCandidate:
    width = max(1.0, float(candidate.width))
    height = max(1.0, float(candidate.height))
    reason = _via_candidate_rejection_reason(width, height, candidate.roundness, settings, expected_span)
    if not reason:
        min_contrast = max(0.0, float(settings.via_min_contrast))
        min_edge_coverage = max(0.0, min(1.0, float(settings.via_min_edge_coverage)))
        coverage = max(0.0, min(1.0, candidate.roundness / 100.0))
        if min_contrast > 0.0 and candidate.contrast < min_contrast:
            reason = "low_contrast"
        elif min_edge_coverage > 0.0 and coverage < min_edge_coverage:
            reason = "low_edge_coverage"
        elif candidate.score < float(settings.via_min_score):
            reason = "low_score"
    return _ViaCandidate(
        center_x=float(candidate.center_x),
        center_y=float(candidate.center_y),
        width=width,
        height=height,
        score=float(candidate.score),
        source=candidate.source,
        roundness=float(candidate.roundness),
        contrast=float(candidate.contrast),
        edge_strength=float(candidate.edge_strength),
        reason=reason,
    )

def _via_candidate_rejection_reason(
    width: float,
    height: float,
    roundness: float,
    settings: ContourExtractionSettings,
    expected_span: int,
) -> str:
    if width <= 0.0 or height <= 0.0:
        return "empty_geometry"
    min_width, max_width, min_height, max_height = _candidate_size_limits(settings, expected_span)
    if width < min_width:
        return "min_via_width"
    if width > max_width:
        return "max_via_width"
    if height < min_height:
        return "min_via_height"
    if height > max_height:
        return "max_via_height"
    aspect = width / max(1.0, height)
    min_aspect = settings.min_aspect_ratio if settings.min_aspect_ratio > 0.0 else 0.25
    max_aspect = settings.max_aspect_ratio if settings.max_aspect_ratio is not None else 4.0
    if aspect < min_aspect:
        return "min_aspect_ratio"
    if aspect > max_aspect:
        return "max_aspect_ratio"
    if roundness < settings.via_min_roundness:
        return "roundness"
    return ""

def _candidate_size_limits(
    settings: ContourExtractionSettings, expected_span: int
) -> tuple[float, float, float, float]:
    if settings.via_size_mode == "fixed" and settings.fixed_via_widths and settings.fixed_via_heights:
        widths = [float(value) for value in settings.fixed_via_widths if int(value) > 0]
        heights = [float(value) for value in settings.fixed_via_heights if int(value) > 0]
        min_width = max(1.0, min(widths) * 0.45)
        max_width = max(widths) * 1.65
        min_height = max(1.0, min(heights) * 0.45)
        max_height = max(heights) * 1.65
        return min_width, max_width, min_height, max_height
    min_width = float(settings.min_via_width if settings.min_via_width > 0 else max(2, expected_span * 0.35))
    min_height = float(settings.min_via_height if settings.min_via_height > 0 else max(2, expected_span * 0.35))
    max_width = float(settings.max_via_width if settings.max_via_width is not None else max(12, expected_span * 3.5))
    max_height = float(settings.max_via_height if settings.max_via_height is not None else max(12, expected_span * 3.5))
    return min_width, max_width, min_height, max_height

def _via_candidate_radii(settings: ContourExtractionSettings, expected_span: int) -> list[int]:
    diameters: list[float] = []
    for width, height in zip(settings.fixed_via_widths, settings.fixed_via_heights, strict=False):
        if int(width) > 0 and int(height) > 0:
            diameters.append((float(width) + float(height)) / 2.0)
    if not diameters:
        min_width, max_width, min_height, max_height = _candidate_size_limits(settings, expected_span)
        min_diameter = max(3.0, min(min_width, min_height, expected_span * 0.75))
        max_diameter = max(min_diameter, min(max_width, max_height, expected_span * 1.55))
        if max_diameter - min_diameter < 2.0:
            diameters.append(float(expected_span))
        else:
            step = max(1.0, expected_span * 0.18)
            value = min_diameter
            while value <= max_diameter + 0.5:
                diameters.append(value)
                value += step
    radii = {max(2, round(diameter / 2.0)) for diameter in diameters}
    if not radii:
        radii.add(max(2, round(expected_span / 2.0)))
    return sorted(radii)

def _iou_nms(
    candidates: list[_ViaCandidate], *, iou_threshold: float
) -> tuple[list[_ViaCandidate], list[_ViaCandidate]]:
    if not candidates:
        return [], []
    accepted: list[_ViaCandidate] = []
    duplicates: list[_ViaCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        is_duplicate = False
        for existing in accepted:
            if _candidate_iou(candidate, existing) >= iou_threshold:
                is_duplicate = True
                break
        if is_duplicate:
            duplicates.append(candidate)
        else:
            accepted.append(candidate)
    accepted.sort(key=lambda item: (item.center_y, item.center_x))
    return accepted, duplicates

def _candidate_iou(first: _ViaCandidate, second: _ViaCandidate) -> float:
    a_left = first.center_x - first.width / 2.0
    a_top = first.center_y - first.height / 2.0
    a_right = a_left + first.width
    a_bottom = a_top + first.height
    b_left = second.center_x - second.width / 2.0
    b_top = second.center_y - second.height / 2.0
    b_right = b_left + second.width
    b_bottom = b_top + second.height
    intersect_left = max(a_left, b_left)
    intersect_top = max(a_top, b_top)
    intersect_right = min(a_right, b_right)
    intersect_bottom = min(a_bottom, b_bottom)
    if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
        return 0.0
    intersection = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
    union = first.width * first.height + second.width * second.height - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)

def _render_via_candidates_mask(
    shape: tuple[int, ...],
    candidates: list[_ViaCandidate],
    settings: ContourExtractionSettings,
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for candidate in candidates:
        box_width, box_height = _rendered_candidate_size(candidate, settings)
        center = (round(candidate.center_x), round(candidate.center_y))
        axes = (max(1, round(box_width / 2.0)), max(1, round(box_height / 2.0)))
        cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255, thickness=-1, lineType=cv2.LINE_8)
    return ensure_binary_mask(mask)

def _rendered_candidate_size(candidate: _ViaCandidate, settings: ContourExtractionSettings) -> tuple[float, float]:
    pairs = list(zip(settings.fixed_via_widths, settings.fixed_via_heights, strict=False))
    if settings.via_size_mode == "fixed" and pairs:
        best_width, best_height = min(
            pairs,
            key=lambda pair: (
                abs(float(pair[0]) - candidate.width) / max(1.0, float(pair[0]))
                + abs(float(pair[1]) - candidate.height) / max(1.0, float(pair[1]))
            ),
        )
        return float(best_width), float(best_height)
    return max(2.0, float(candidate.width)), max(2.0, float(candidate.height))
