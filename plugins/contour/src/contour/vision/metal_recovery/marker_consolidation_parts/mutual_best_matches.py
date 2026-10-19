"""marker_consolidation: mutual best matches."""
from __future__ import annotations
from typing import TYPE_CHECKING

from contour.vision.metal_recovery.marker_consolidation_parts.module_state import (
    _AMBIGUITY_RATIO,
    _MIN_CONTINUATION_SCORE,
)

from contour.vision.metal_recovery.marker_consolidation_parts.reason import (
    _endpoint_key_a,
    _endpoint_key_b,
    _endpoint_pair_key,
    _with_reason,
)

def _mutual_best_matches(
    scored: list[RidgeLink],
    *,
    endpoint_count: int,
) -> tuple[tuple[RidgeLink, ...], tuple[RidgeLink, ...], dict[str, int]]:
    del endpoint_count
    reasons: dict[str, int] = {}
    rejected: list[RidgeLink] = []
    eligible_by_endpoint: dict[tuple[int, int], list[RidgeLink]] = {}
    for link in scored:
        if link.reason:
            reasons[link.reason] = reasons.get(link.reason, 0) + 1
            if link.score >= 0.35 or link.reason == "boundary_veto":
                rejected.append(link)
            continue
        eligible_by_endpoint.setdefault(_endpoint_key_a(link), []).append(link)
        eligible_by_endpoint.setdefault(_endpoint_key_b(link), []).append(link)

    best: dict[tuple[int, int], RidgeLink] = {}
    ambiguous_endpoints: set[tuple[int, int]] = set()
    for endpoint_key, links in eligible_by_endpoint.items():
        ranked = sorted(links, key=lambda item: item.score, reverse=True)
        if (
            len(ranked) >= 2
            and ranked[1].score >= _AMBIGUITY_RATIO * max(ranked[0].score, 1e-6)
            and ranked[1].score >= _MIN_CONTINUATION_SCORE
        ):
            ambiguous_endpoints.add(endpoint_key)
            reasons["ambiguous_competing_candidate"] = (
                reasons.get("ambiguous_competing_candidate", 0) + 1
            )
            for item in ranked[:2]:
                rejected.append(_with_reason(item, "ambiguous_competing_candidate"))
            continue
        best[endpoint_key] = ranked[0]

    accepted: list[RidgeLink] = []
    used_endpoints: set[tuple[int, int]] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for endpoint_key, link in best.items():
        other_key = _endpoint_key_b(link) if endpoint_key == _endpoint_key_a(link) else _endpoint_key_a(link)
        if endpoint_key in ambiguous_endpoints or other_key in ambiguous_endpoints:
            continue
        partner_best = best.get(other_key)
        if partner_best is None:
            continue
        if _endpoint_pair_key(partner_best) != _endpoint_pair_key(link):
            continue
        if endpoint_key in used_endpoints or other_key in used_endpoints:
            continue
        fragment_pair = (
            min(link.fragment_a, link.fragment_b),
            max(link.fragment_a, link.fragment_b),
        )
        if fragment_pair in seen_pairs:
            continue
        accepted.append(link)
        used_endpoints.add(endpoint_key)
        used_endpoints.add(other_key)
        seen_pairs.add(fragment_pair)

    return tuple(accepted), tuple(rejected), reasons

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import RidgeLink
