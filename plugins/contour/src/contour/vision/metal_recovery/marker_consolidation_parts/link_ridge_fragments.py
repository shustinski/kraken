"""marker_consolidation: link ridge fragments."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from contour.vision.metal_recovery.marker_consolidation_parts.module_state import _BOUNDARY_PERCENTILE

from contour.vision.metal_recovery.marker_consolidation_parts.extract_ridge_fragments import _extract_ridge_fragments

from contour.vision.metal_recovery.marker_consolidation_parts.fragment_endpoints import _score_endpoint_pairs

from contour.vision.metal_recovery.marker_consolidation_parts.mutual_best_matches import _mutual_best_matches

from contour.vision.metal_recovery.marker_consolidation_parts.render_debug import (
    _remap_by_parent,
    _union,
)

def _link_ridge_fragments(
    ridge_labels: np.ndarray,
    ridge_count: int,
    stats: np.ndarray,
    centroids: np.ndarray,
    evidence: ConsolidationEvidence,
    *,
    orientation_aware_veto: bool = False,
) -> tuple[np.ndarray, tuple[RidgeLink, ...], tuple[RidgeLink, ...], dict[str, int]]:
    fragments = _extract_ridge_fragments(
        ridge_labels,
        ridge_count,
        stats,
        centroids,
        evidence,
    )
    endpoints = tuple(endpoint for fragment in fragments for endpoint in fragment.endpoints)
    if len(endpoints) < 2:
        return ridge_labels.copy(), (), (), {}

    edge_veto = float(np.percentile(evidence.persistent_edge, _BOUNDARY_PERCENTILE))
    edge_veto = max(edge_veto, 1e-3)
    scored = _score_endpoint_pairs(
        endpoints,
        evidence,
        edge_veto,
        orientation_aware_veto=orientation_aware_veto,
    )
    accepted, rejected, reasons = _mutual_best_matches(scored, endpoint_count=len(endpoints))
    parent = np.arange(ridge_count + 1, dtype=np.int32)
    for link in accepted:
        _union(parent, link.fragment_a, link.fragment_b)
    logical = _remap_by_parent(ridge_labels, parent)
    return logical, accepted, rejected, reasons

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence, RidgeLink
