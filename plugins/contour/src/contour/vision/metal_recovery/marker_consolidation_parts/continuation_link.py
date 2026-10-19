"""marker_consolidation: continuation link."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from contour.vision.metal_recovery.marker_consolidation_parts.module_state import (
    _CONE_COSINE,
    _MAX_ANGLE_DELTA,
    _MAX_TRANSVERSE_OFFSET_PX,
    _MIN_CONTINUATION_SCORE,
    _SHORT_GAP_PX,
    _WEIGHTS,
)

from contour.vision.metal_recovery.marker_consolidation_parts.corridor_boundary_veto import _corridor_boundary_veto

from contour.vision.metal_recovery.marker_consolidation_parts.render_debug import _acute_angle_delta

from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import RidgeLink

from contour.vision.metal_recovery.marker_consolidation_parts.sample_corridor import _sample_corridor

def _continuation_link(
    source: _Endpoint,
    target: _Endpoint,
    evidence: ConsolidationEvidence,
    edge_veto: float,
    *,
    dy: float,
    dx: float,
    orientation_aware_veto: bool = False,
) -> RidgeLink | None:
    length = float(np.hypot(dx, dy))
    direction_y = dy / length
    direction_x = dx / length
    forward = direction_x * source.tangent_x + direction_y * source.tangent_y
    backward = (-direction_x) * target.tangent_x + (-direction_y) * target.tangent_y
    if forward < _CONE_COSINE or backward < _CONE_COSINE:
        return None

    source_angle = float(np.arctan2(source.tangent_y, source.tangent_x))
    target_angle = float(np.arctan2(target.tangent_y, target.tangent_x))
    angle_delta = _acute_angle_delta(source_angle, target_angle)
    orientation_score = max(0.0, 1.0 - angle_delta / _MAX_ANGLE_DELTA)
    longitudinal = min(forward, backward)
    transverse = length * float(np.sqrt(max(0.0, 1.0 - longitudinal * longitudinal)))
    width_limit = max(_MAX_TRANSVERSE_OFFSET_PX, 0.6 * (source.width + target.width))
    offset_penalty = min(1.0, transverse / max(width_limit, 1e-3))

    corridor = _sample_corridor(
        source.row,
        source.col,
        target.row,
        target.col,
        evidence,
    )
    if corridor is None:
        return None
    interior = corridor.center_interior
    if not np.any(interior):
        interior = np.ones(corridor.center_intensity.shape[0], dtype=bool)
    ridge_mean = float(np.mean(corridor.center_ridge[interior]))
    intensity_mean = float(np.mean(corridor.center_intensity[interior]))
    edge_peak = float(np.max(corridor.center_edge[interior]))
    crossing_peak = float(np.max(corridor.center_crossing[interior]))
    fragment_ridge = max(0.5 * (source.mean_ridge + target.mean_ridge), 1e-3)
    fragment_intensity = 0.5 * (source.mean_intensity + target.mean_intensity)
    ridge_score = min(1.0, ridge_mean / fragment_ridge)
    intensity_score = 1.0 - min(1.0, abs(intensity_mean - fragment_intensity) / 35.0)
    conductor = (
        (corridor.center_intensity >= fragment_intensity - 18.0)
        | (corridor.center_ridge >= 0.45 * fragment_ridge)
    )
    corridor_score = float(np.mean(conductor[interior]))
    if orientation_aware_veto:
        boundary_score = min(1.0, crossing_peak)
    else:
        boundary_score = min(1.0, edge_peak / edge_veto)

    reason = ""
    if angle_delta > _MAX_ANGLE_DELTA:
        reason = "angle_mismatch"
    elif transverse > width_limit:
        reason = "transverse_offset"
    elif length > _SHORT_GAP_PX and _corridor_boundary_veto(
        evidence,
        source,
        target,
        edge_veto,
        edge_peak=edge_peak,
        orientation_aware=orientation_aware_veto,
    ):
        reason = "boundary_veto"
    elif corridor_score < 0.28 and length > _SHORT_GAP_PX:
        reason = "corridor_too_weak"

    score = (
        _WEIGHTS["angle"] * orientation_score
        + _WEIGHTS["parallel"] * longitudinal
        + _WEIGHTS["corridor"] * corridor_score
        + _WEIGHTS["ridge"] * ridge_score
        + _WEIGHTS["intensity"] * intensity_score
        - _WEIGHTS["boundary"] * boundary_score
        - _WEIGHTS["offset"] * offset_penalty
    )
    score = float(np.clip(score, 0.0, 1.0))
    if reason == "" and score < _MIN_CONTINUATION_SCORE and length > _SHORT_GAP_PX:
        reason = "corridor_too_weak"
    return RidgeLink(
        fragment_a=source.fragment_id,
        fragment_b=target.fragment_id,
        endpoint_a=source.index,
        endpoint_b=target.index,
        point_a=(source.col, source.row),
        point_b=(target.col, target.row),
        score=score,
        orientation_score=orientation_score,
        longitudinal_score=float(longitudinal),
        corridor_score=corridor_score,
        ridge_score=ridge_score,
        intensity_score=intensity_score,
        boundary_score=boundary_score,
        offset_penalty=offset_penalty,
        reason=reason,
    )

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence, _Endpoint
