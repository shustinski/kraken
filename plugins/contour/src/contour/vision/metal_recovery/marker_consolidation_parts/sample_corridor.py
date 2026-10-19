"""marker_consolidation: sample corridor."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from contour.vision.metal_recovery.marker_consolidation_parts.module_state import _CORRIDOR_HALF_WIDTH_PX

from contour.vision.metal_recovery.marker_consolidation_parts.corridor_boundary_veto import _CorridorSample

def _sample_corridor(
    row_a: int,
    col_a: int,
    row_b: int,
    col_b: int,
    evidence: ConsolidationEvidence,
) -> _CorridorSample | None:
    length = max(round(np.hypot(col_b - col_a, row_b - row_a)), 1)
    rows = np.linspace(row_a, row_b, length + 1)
    cols = np.linspace(col_a, col_b, length + 1)
    dy = float(row_b - row_a)
    dx = float(col_b - col_a)
    norm = max(float(np.hypot(dx, dy)), 1e-6)
    dir_x = dx / norm
    dir_y = dy / norm
    nx = -dir_y
    ny = dir_x
    height, width = evidence.intensity.shape
    sampled_edge: list[np.ndarray] = []
    center_intensity: np.ndarray | None = None
    center_ridge: np.ndarray | None = None
    center_edge: np.ndarray | None = None
    center_crossing: np.ndarray | None = None
    for offset in (-_CORRIDOR_HALF_WIDTH_PX, 0.0, _CORRIDOR_HALF_WIDTH_PX):
        yy = np.clip(np.round(rows + offset * ny).astype(np.int32), 0, height - 1)
        xx = np.clip(np.round(cols + offset * nx).astype(np.int32), 0, width - 1)
        edge = evidence.persistent_edge[yy, xx].astype(np.float32)
        sampled_edge.append(edge)
        if offset == 0.0:
            center_intensity = evidence.intensity[yy, xx].astype(np.float32)
            center_ridge = evidence.ridge_confidence[yy, xx].astype(np.float32)
            center_edge = edge
            gx = evidence.gradient_x[yy, xx].astype(np.float32)
            gy = evidence.gradient_y[yy, xx].astype(np.float32)
            magnitude = np.maximum(evidence.magnitude[yy, xx].astype(np.float32), 1e-3)
            center_crossing = np.abs(gx * dir_x + gy * dir_y) / magnitude
    if (
        center_intensity is None
        or center_ridge is None
        or center_edge is None
        or center_crossing is None
    ):
        return None
    n = int(center_intensity.size)
    center_interior = np.ones(n, dtype=bool)
    if n >= 5:
        trim = max(1, round(0.15 * n))
        center_interior[:trim] = False
        center_interior[-trim:] = False
    edge = np.concatenate(sampled_edge)
    interior = np.concatenate([center_interior, center_interior, center_interior])
    return _CorridorSample(
        center_intensity=center_intensity,
        center_ridge=center_ridge,
        center_edge=center_edge,
        center_crossing=center_crossing,
        center_interior=center_interior,
        edge=edge,
        interior=interior,
    )

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence
