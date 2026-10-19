"""marker_consolidation: corridor boundary veto."""
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from contour.vision.metal_recovery.conductor_bands import (
    corridor_has_transverse_separator,
    evidence_from_consolidation,
)

def _corridor_boundary_veto(
    evidence: ConsolidationEvidence,
    source: _Endpoint,
    target: _Endpoint,
    edge_veto: float,
    *,
    edge_peak: float,
    orientation_aware: bool,
) -> bool:
    if not orientation_aware:
        return edge_peak >= edge_veto
    return corridor_has_transverse_separator(
        evidence_from_consolidation(evidence),
        source.row,
        source.col,
        target.row,
        target.col,
        edge_veto=edge_veto,
    )

@dataclass(frozen=True, slots=True)
class _CorridorSample:
    center_intensity: np.ndarray
    center_ridge: np.ndarray
    center_edge: np.ndarray
    center_crossing: np.ndarray
    center_interior: np.ndarray
    edge: np.ndarray
    interior: np.ndarray

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence, _Endpoint
