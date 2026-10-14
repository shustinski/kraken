"""marker_consolidation: reason."""
from __future__ import annotations

from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import RidgeLink

def _endpoint_key_a(link: RidgeLink) -> tuple[int, int]:
    return (link.fragment_a, link.endpoint_a)

def _endpoint_key_b(link: RidgeLink) -> tuple[int, int]:
    return (link.fragment_b, link.endpoint_b)

def _endpoint_pair_key(link: RidgeLink) -> tuple[tuple[int, int], tuple[int, int]]:
    first = _endpoint_key_a(link)
    second = _endpoint_key_b(link)
    return (first, second) if first <= second else (second, first)

def _with_reason(link: RidgeLink, reason: str) -> RidgeLink:
    return RidgeLink(
        fragment_a=link.fragment_a,
        fragment_b=link.fragment_b,
        endpoint_a=link.endpoint_a,
        endpoint_b=link.endpoint_b,
        point_a=link.point_a,
        point_b=link.point_b,
        score=link.score,
        orientation_score=link.orientation_score,
        longitudinal_score=link.longitudinal_score,
        corridor_score=link.corridor_score,
        ridge_score=link.ridge_score,
        intensity_score=link.intensity_score,
        boundary_score=link.boundary_score,
        offset_penalty=link.offset_penalty,
        reason=reason,
    )
