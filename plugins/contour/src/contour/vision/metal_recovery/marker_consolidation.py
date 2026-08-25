"""Instance-aware consolidation of structural marker fragments.

Ridge fragments are linked only as longitudinal same-trace continuations.
Wide-interior fragments are merged only when they sit in one enclosed
conductor basin without a separating boundary.  Parallel conductors stay
distinct: a strong transverse offset or transverse boundary vetoes a link.

Ground truth, frame ids, and filenames are never consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from ...utils import ensure_binary_mask, ensure_uint8
from .conductor_bands import (
    corridor_has_transverse_separator,
    evidence_from_consolidation,
    group_ridges_by_conductor_band,
    region_has_internal_separator,
)

RejectionReason = Literal[
    "angle_mismatch",
    "transverse_offset",
    "boundary_veto",
    "corridor_too_weak",
    "ambiguous_competing_candidate",
]

_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
_SEARCH_RADIUS_PX = 16.0
_CONE_COSINE = 0.82
_MAX_ANGLE_DELTA = np.deg2rad(25.0)
_MAX_TRANSVERSE_OFFSET_PX = 3.0
_CORRIDOR_HALF_WIDTH_PX = 1.5
_MIN_CONTINUATION_SCORE = 0.55
_SHORT_GAP_PX = 3.0
_AMBIGUITY_RATIO = 0.90
_BOUNDARY_PERCENTILE = 80.0
_LOCAL_WIDE_RADIUS_PX = 8.0
_WEIGHTS = {
    "angle": 0.22,
    "parallel": 0.22,
    "corridor": 0.18,
    "ridge": 0.14,
    "intensity": 0.12,
    "boundary": 0.45,
    "offset": 0.35,
}


@dataclass(frozen=True, slots=True)
class ConsolidationEvidence:
    """Image evidence used by consolidation. No GT, ids, or filenames."""

    intensity: np.ndarray
    ridge_confidence: np.ndarray
    ridge_orientation: np.ndarray
    structure_orientation: np.ndarray
    coherence: np.ndarray
    persistent_edge: np.ndarray
    magnitude: np.ndarray
    rim_response: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray


@dataclass(frozen=True, slots=True)
class RidgeLink:
    fragment_a: int
    fragment_b: int
    endpoint_a: int
    endpoint_b: int
    point_a: tuple[int, int]
    point_b: tuple[int, int]
    score: float
    orientation_score: float
    longitudinal_score: float
    corridor_score: float
    ridge_score: float
    intensity_score: float
    boundary_score: float
    offset_penalty: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MarkerConsolidationStats:
    raw_ridge_count: int = 0
    logical_ridge_count: int = 0
    accepted_ridge_links: int = 0
    rejected_ridge_links: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    raw_wide_count: int = 0
    logical_wide_count: int = 0
    wide_unions: int = 0
    combined_logical_count: int = 0
    band_count: int = 0
    accepted_band_groups: int = 0
    wide_kept_with_multi_ridge: int = 0
    wide_dropped_for_separator: int = 0


@dataclass(frozen=True, slots=True)
class MarkerConsolidationResult:
    ridge_raw_labels: np.ndarray
    ridge_logical_labels: np.ndarray
    wide_raw_labels: np.ndarray
    wide_logical_labels: np.ndarray
    combined_labels: np.ndarray
    accepted_links: tuple[RidgeLink, ...]
    rejected_links: tuple[RidgeLink, ...]
    stats: MarkerConsolidationStats
    debug_images: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class _Endpoint:
    fragment_id: int
    index: int
    row: int
    col: int
    tangent_y: float
    tangent_x: float
    width: float
    mean_intensity: float
    mean_ridge: float


@dataclass(frozen=True, slots=True)
class _Fragment:
    fragment_id: int
    area: int
    centroid_row: float
    centroid_col: float
    mean_intensity: float
    mean_ridge: float
    width: float
    endpoints: tuple[_Endpoint, ...]


def consolidate_markers(
    ridge_mask: np.ndarray,
    wide_mask: np.ndarray,
    evidence: ConsolidationEvidence,
    *,
    link_ridge: bool,
    link_wide: bool,
    group_bands: bool = False,
    orientation_aware_veto: bool = False,
    separator_aware_combine: bool = False,
    min_marker_area: int = 16,
    wide_interior_radius: float = 4.0,
) -> MarkerConsolidationResult:
    ridge_count, ridge_raw, ridge_stats, ridge_centroids = _connected_with_stats(ridge_mask)
    wide_count, wide_raw, wide_stats, wide_centroids = _connected_with_stats(wide_mask)
    accepted: tuple[RidgeLink, ...] = ()
    rejected: tuple[RidgeLink, ...] = ()
    reasons: dict[str, int] = {}

    if link_ridge and ridge_count > 0:
        ridge_logical, accepted, rejected, reasons = _link_ridge_fragments(
            ridge_raw,
            ridge_count,
            ridge_stats,
            ridge_centroids,
            evidence,
            orientation_aware_veto=orientation_aware_veto,
        )
    else:
        ridge_logical = ridge_raw.copy()

    band_debug = _empty_band_images(ridge_raw.shape)
    band_count = 0
    accepted_band_groups = 0
    if group_bands and np.any(ridge_logical):
        grouped, band_stats, band_debug = group_ridges_by_conductor_band(
            ridge_logical,
            evidence_from_consolidation(evidence),
        )
        ridge_logical = grouped
        band_count = int(band_stats.band_count)
        accepted_band_groups = int(band_stats.accepted_band_groups)

    if link_wide and wide_count > 0:
        wide_logical, wide_unions = _consolidate_wide_fragments(
            wide_raw,
            wide_count,
            wide_stats,
            wide_centroids,
            ridge_raw,
            evidence,
            min_marker_area=min_marker_area,
            wide_interior_radius=wide_interior_radius,
        )
    else:
        wide_logical = wide_raw.copy()
        wide_unions = 0

    combined, kept_multi, dropped_sep = _combine_logical_markers(
        ridge_logical,
        wide_logical,
        evidence,
        separator_aware=separator_aware_combine,
    )
    stats = MarkerConsolidationStats(
        raw_ridge_count=ridge_count,
        logical_ridge_count=_positive_id_count(ridge_logical),
        accepted_ridge_links=len(accepted),
        rejected_ridge_links=len(rejected),
        rejection_reasons=reasons,
        raw_wide_count=wide_count,
        logical_wide_count=_positive_id_count(wide_logical),
        wide_unions=wide_unions,
        combined_logical_count=_positive_id_count(combined),
        band_count=band_count,
        accepted_band_groups=accepted_band_groups,
        wide_kept_with_multi_ridge=kept_multi,
        wide_dropped_for_separator=dropped_sep,
    )
    debug = _render_debug(
        evidence.intensity,
        ridge_raw=ridge_raw,
        ridge_logical=ridge_logical,
        wide_raw=wide_raw,
        wide_logical=wide_logical,
        combined=combined,
        accepted=accepted,
        rejected=rejected,
    )
    debug.update(band_debug)
    return MarkerConsolidationResult(
        ridge_raw_labels=ridge_raw,
        ridge_logical_labels=ridge_logical,
        wide_raw_labels=wide_raw,
        wide_logical_labels=wide_logical,
        combined_labels=combined,
        accepted_links=accepted,
        rejected_links=rejected,
        stats=stats,
        debug_images=debug,
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


def _extract_ridge_fragments(
    ridge_labels: np.ndarray,
    ridge_count: int,
    stats: np.ndarray,
    centroids: np.ndarray,
    evidence: ConsolidationEvidence,
) -> tuple[_Fragment, ...]:
    binary = (ridge_labels > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(
        binary.astype(np.float32),
        cv2.CV_32F,
        _NEIGHBOR_KERNEL.astype(np.float32),
    )
    fragments: list[_Fragment] = []
    height, width = ridge_labels.shape
    for fragment_id in range(1, ridge_count + 1):
        left = int(stats[fragment_id, cv2.CC_STAT_LEFT])
        top = int(stats[fragment_id, cv2.CC_STAT_TOP])
        box_w = int(stats[fragment_id, cv2.CC_STAT_WIDTH])
        box_h = int(stats[fragment_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[fragment_id, cv2.CC_STAT_AREA])
        y0 = max(0, top - 1)
        x0 = max(0, left - 1)
        y1 = min(height, top + box_h + 1)
        x1 = min(width, left + box_w + 1)
        roi = ridge_labels[y0:y1, x0:x1] == fragment_id
        if not np.any(roi):
            continue
        mean_intensity = float(np.mean(evidence.intensity[y0:y1, x0:x1][roi]))
        mean_ridge = float(np.mean(evidence.ridge_confidence[y0:y1, x0:x1][roi]))
        distance = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 3)
        width_px = max(float(distance.max()), 1.0)
        centroid_row = float(centroids[fragment_id, 1])
        centroid_col = float(centroids[fragment_id, 0])
        endpoints = _fragment_endpoints(
            fragment_id=fragment_id,
            roi=roi,
            origin_row=y0,
            origin_col=x0,
            neighbor_roi=neighbor_count[y0:y1, x0:x1],
            evidence=evidence,
            centroid_row=centroid_row,
            centroid_col=centroid_col,
            width_px=width_px,
            mean_intensity=mean_intensity,
            mean_ridge=mean_ridge,
        )
        fragments.append(
            _Fragment(
                fragment_id=fragment_id,
                area=area,
                centroid_row=centroid_row,
                centroid_col=centroid_col,
                mean_intensity=mean_intensity,
                mean_ridge=mean_ridge,
                width=width_px,
                endpoints=endpoints,
            )
        )
    return tuple(fragments)


def _fragment_endpoints(
    *,
    fragment_id: int,
    roi: np.ndarray,
    origin_row: int,
    origin_col: int,
    neighbor_roi: np.ndarray,
    evidence: ConsolidationEvidence,
    centroid_row: float,
    centroid_col: float,
    width_px: float,
    mean_intensity: float,
    mean_ridge: float,
) -> tuple[_Endpoint, ...]:
    local_ends = np.argwhere(roi & (neighbor_roi <= 1))
    if local_ends.shape[0] > 2:
        local_ends = _two_farthest_points(local_ends)
    if local_ends.shape[0] < 2:
        local_ends = _principal_extremes(roi)
    endpoints: list[_Endpoint] = []
    for index, (local_row, local_col) in enumerate(local_ends[:2]):
        row = int(origin_row + local_row)
        col = int(origin_col + local_col)
        tangent_y, tangent_x = _outward_tangent(
            evidence,
            row=row,
            col=col,
            centroid_row=centroid_row,
            centroid_col=centroid_col,
        )
        endpoints.append(
            _Endpoint(
                fragment_id=fragment_id,
                index=index,
                row=row,
                col=col,
                tangent_y=tangent_y,
                tangent_x=tangent_x,
                width=width_px,
                mean_intensity=mean_intensity,
                mean_ridge=mean_ridge,
            )
        )
    return tuple(endpoints)


def _two_farthest_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 2:
        return points
    coords = points.astype(np.float32)
    delta = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(delta * delta, axis=2)
    first, second = np.unravel_index(int(np.argmax(dist_sq)), dist_sq.shape)
    return points[(first, second), :]


def _principal_extremes(roi: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(roi)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if rows.size == 1:
        return np.array([[rows[0], cols[0]]], dtype=np.int32)
    coords = np.column_stack((cols.astype(np.float32), rows.astype(np.float32)))
    centered = coords - coords.mean(axis=0, keepdims=True)
    gram = centered.T @ centered
    _values, vectors = np.linalg.eigh(gram)
    axis = vectors[:, -1]
    projection = centered @ axis
    return np.array(
        [
            [int(rows[int(np.argmin(projection))]), int(cols[int(np.argmin(projection))])],
            [int(rows[int(np.argmax(projection))]), int(cols[int(np.argmax(projection))])],
        ],
        dtype=np.int32,
    )


def _outward_tangent(
    evidence: ConsolidationEvidence,
    *,
    row: int,
    col: int,
    centroid_row: float,
    centroid_col: float,
) -> tuple[float, float]:
    height, width = evidence.ridge_orientation.shape
    y0 = max(0, row - 2)
    x0 = max(0, col - 2)
    y1 = min(height, row + 3)
    x1 = min(width, col + 3)
    patch = evidence.ridge_orientation[y0:y1, x0:x1]
    angle = float(np.median(patch)) if patch.size else 0.0
    tangent_x = float(np.cos(angle))
    tangent_y = float(np.sin(angle))
    away_x = float(col) - centroid_col
    away_y = float(row) - centroid_row
    if tangent_x * away_x + tangent_y * away_y < 0.0:
        tangent_x = -tangent_x
        tangent_y = -tangent_y
    norm = max(float(np.hypot(tangent_x, tangent_y)), 1e-6)
    return tangent_y / norm, tangent_x / norm


def _score_endpoint_pairs(
    endpoints: tuple[_Endpoint, ...],
    evidence: ConsolidationEvidence,
    edge_veto: float,
    *,
    orientation_aware_veto: bool = False,
) -> list[RidgeLink]:
    buckets = _spatial_buckets(endpoints, cell=_SEARCH_RADIUS_PX)
    scored: list[RidgeLink] = []
    radius_sq = _SEARCH_RADIUS_PX * _SEARCH_RADIUS_PX
    for index, source in enumerate(endpoints):
        cell_col = int(source.col // _SEARCH_RADIUS_PX)
        cell_row = int(source.row // _SEARCH_RADIUS_PX)
        for neighbor_index in _bucket_neighbors(buckets, cell_col, cell_row):
            if neighbor_index <= index:
                continue
            target = endpoints[neighbor_index]
            if target.fragment_id == source.fragment_id:
                continue
            dy = float(target.row - source.row)
            dx = float(target.col - source.col)
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1.0 or dist_sq > radius_sq:
                continue
            link = _continuation_link(
                source,
                target,
                evidence,
                edge_veto,
                dy=dy,
                dx=dx,
                orientation_aware_veto=orientation_aware_veto,
            )
            if link is None:
                continue
            scored.append(link)
    return scored


def _spatial_buckets(endpoints: tuple[_Endpoint, ...], *, cell: float) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, endpoint in enumerate(endpoints):
        key = (int(endpoint.col // cell), int(endpoint.row // cell))
        buckets.setdefault(key, []).append(index)
    return buckets


def _bucket_neighbors(
    buckets: dict[tuple[int, int], list[int]],
    cell_col: int,
    cell_row: int,
) -> tuple[int, ...]:
    found: list[int] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            found.extend(buckets.get((cell_col + dx, cell_row + dy), ()))
    return tuple(found)


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


def _sample_corridor(
    row_a: int,
    col_a: int,
    row_b: int,
    col_b: int,
    evidence: ConsolidationEvidence,
) -> _CorridorSample | None:
    length = max(int(round(np.hypot(col_b - col_a, row_b - row_a))), 1)
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
        trim = max(1, int(round(0.15 * n)))
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


def _consolidate_wide_fragments(
    wide_labels: np.ndarray,
    wide_count: int,
    wide_stats: np.ndarray,
    wide_centroids: np.ndarray,
    ridge_labels: np.ndarray,
    evidence: ConsolidationEvidence,
    *,
    min_marker_area: int,
    wide_interior_radius: float,
) -> tuple[np.ndarray, int]:
    basins = _enclosed_conductor_basins(
        evidence,
        min_marker_area=min_marker_area,
        wide_interior_radius=wide_interior_radius,
    )
    parent = np.arange(wide_count + 1, dtype=np.int32)
    fragment_basin = _fragment_majority_labels(wide_labels, wide_count, basins)
    basin_to_fragments: dict[int, list[int]] = {}
    basin_blocked: dict[int, bool] = {}
    for fragment_id in range(1, wide_count + 1):
        basin_id = int(fragment_basin[fragment_id])
        if basin_id <= 0:
            continue
        blocked = basin_blocked.get(basin_id)
        if blocked is None:
            blocked = _basin_has_parallel_ridges(basins == basin_id, ridge_labels)
            basin_blocked[basin_id] = blocked
        if blocked:
            continue
        basin_to_fragments.setdefault(basin_id, []).append(fragment_id)

    unions = 0
    for members in basin_to_fragments.values():
        if len(members) < 2:
            continue
        root = members[0]
        for other in members[1:]:
            if _find(parent, root) != _find(parent, other):
                _union(parent, root, other)
                unions += 1

    unions += _local_wide_unions(
        wide_count,
        wide_stats,
        wide_centroids,
        parent,
        fragment_basin,
        evidence,
    )
    logical = _remap_by_parent(wide_labels, parent)
    return logical, unions


def _enclosed_conductor_basins(
    evidence: ConsolidationEvidence,
    *,
    min_marker_area: int,
    wide_interior_radius: float,
) -> np.ndarray:
    radius = max(1.0, float(wide_interior_radius))
    rim_limit = float(np.percentile(evidence.rim_response, 85.0))
    walls = np.where(evidence.rim_response >= max(rim_limit, 1e-4), 255, 0).astype(np.uint8)
    walls = cv2.dilate(walls, np.ones((5, 5), np.uint8))
    open_space = np.where(walls > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    if count <= 1:
        return np.zeros(open_space.shape, dtype=np.int32)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    box_perimeter = np.maximum(2.0 * (widths + heights), 1.0)
    dilated = cv2.dilate(walls, np.ones((3, 3), np.uint8))
    contact = np.bincount(
        labels.ravel(),
        weights=(dilated > 0).ravel().astype(np.float64),
        minlength=count,
    )
    gradient_means = np.bincount(
        labels.ravel(),
        weights=evidence.magnitude.astype(np.float64).ravel(),
        minlength=count,
    )
    pixel_counts = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
    gradient_means = np.divide(gradient_means, np.maximum(pixel_counts, 1.0))
    gradient_limit = max(float(np.percentile(evidence.magnitude, 60.0)), 8.0)
    keep = (
        (areas >= max(float(min_marker_area), 16.0 * radius * radius))
        & (areas <= 0.45 * float(open_space.size))
        & (contact >= 0.55 * box_perimeter)
        & (np.minimum(widths, heights) >= 2.0 * radius)
        & (gradient_means <= gradient_limit)
    )
    keep[0] = False
    return np.where(keep[labels], labels, 0).astype(np.int32)


def _basin_has_parallel_ridges(basin: np.ndarray, ridge_labels: np.ndarray) -> bool:
    """True only if two distinct ridges live inside the basin, not merely graze the rim."""

    interior = cv2.erode(basin.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if not np.any(interior) or not np.any(ridge_labels):
        return False
    ridge_max = int(ridge_labels.max())
    total = np.bincount(ridge_labels.ravel(), minlength=ridge_max + 1)
    inside = np.bincount(ridge_labels[interior].ravel(), minlength=ridge_max + 1)
    lives_inside = (inside[1:] * 2 >= total[1:]) & (inside[1:] > 0)
    return int(np.count_nonzero(lives_inside)) >= 2


def _local_wide_unions(
    wide_count: int,
    stats: np.ndarray,
    centroids: np.ndarray,
    parent: np.ndarray,
    fragment_basin: np.ndarray,
    evidence: ConsolidationEvidence,
) -> int:
    unassigned = [
        fragment_id
        for fragment_id in range(1, wide_count + 1)
        if int(fragment_basin[fragment_id]) <= 0
    ]
    if len(unassigned) < 2:
        return 0
    cell = max(_LOCAL_WIDE_RADIUS_PX, 1.0)
    buckets: dict[tuple[int, int], list[int]] = {}
    for fragment_id in unassigned:
        key = (
            int(centroids[fragment_id, 0] // cell),
            int(centroids[fragment_id, 1] // cell),
        )
        buckets.setdefault(key, []).append(fragment_id)
    edge_veto = max(float(np.percentile(evidence.persistent_edge, 70.0)), 1e-3)
    unions = 0
    seen: set[tuple[int, int]] = set()
    for fragment_id in unassigned:
        col = int(centroids[fragment_id, 0] // cell)
        row = int(centroids[fragment_id, 1] // cell)
        c1x = float(centroids[fragment_id, 0])
        c1y = float(centroids[fragment_id, 1])
        r1 = 0.5 * max(
            float(stats[fragment_id, cv2.CC_STAT_WIDTH]),
            float(stats[fragment_id, cv2.CC_STAT_HEIGHT]),
        )
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for other in buckets.get((col + dx, row + dy), ()):
                    if other <= fragment_id:
                        continue
                    pair = (fragment_id, other)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if _find(parent, fragment_id) == _find(parent, other):
                        continue
                    c2x = float(centroids[other, 0])
                    c2y = float(centroids[other, 1])
                    dist = float(np.hypot(c2x - c1x, c2y - c1y))
                    r2 = 0.5 * max(
                        float(stats[other, cv2.CC_STAT_WIDTH]),
                        float(stats[other, cv2.CC_STAT_HEIGHT]),
                    )
                    if dist > min(_LOCAL_WIDE_RADIUS_PX, 0.9 * (r1 + r2) + 3.0):
                        continue
                    sample = _sample_corridor(int(c1y), int(c1x), int(c2y), int(c2x), evidence)
                    if sample is None or not np.any(sample.interior):
                        continue
                    if float(np.max(sample.edge[sample.interior])) >= edge_veto:
                        continue
                    _union(parent, fragment_id, other)
                    unions += 1
    return unions


def _fragment_majority_labels(
    component_labels: np.ndarray,
    component_count: int,
    region_labels: np.ndarray,
) -> np.ndarray:
    assigned = np.zeros(component_count + 1, dtype=np.int32)
    if component_count <= 0 or not np.any(component_labels):
        return assigned
    region_max = int(max(int(region_labels.max()), 0))
    stride = region_max + 1
    if stride <= 1:
        return assigned
    valid = (component_labels > 0) & (region_labels > 0)
    if not np.any(valid):
        return assigned
    encoded = component_labels.astype(np.int64) * stride + region_labels.astype(np.int64)
    counts = np.bincount(encoded[valid], minlength=stride * (component_count + 1))
    for fragment_id in range(1, component_count + 1):
        start = fragment_id * stride
        chunk = counts[start : start + stride]
        if chunk.size == 0:
            continue
        best = int(np.argmax(chunk))
        if best > 0 and int(chunk[best]) > 0:
            assigned[fragment_id] = best
    return assigned


def _combine_logical_markers(
    ridge_labels: np.ndarray,
    wide_labels: np.ndarray,
    evidence: ConsolidationEvidence | None = None,
    *,
    separator_aware: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Keep a wide region unless a real interior separator splits it.

    Ridge count inside the region is evidence, not a reason to drop the plate.
    """

    if not np.any(ridge_labels):
        return _compact_labels(wide_labels), 0, 0
    if not np.any(wide_labels):
        return _compact_labels(ridge_labels), 0, 0

    ridge_max = int(ridge_labels.max())
    wide_max = int(wide_labels.max())
    keep_wide = np.ones(wide_max + 1, dtype=bool)
    keep_wide[0] = False
    eroded = cv2.erode((wide_labels > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    interior = (ridge_labels > 0) & (eroded > 0)
    ridge_hits = np.zeros(wide_max + 1, dtype=np.int32)
    if np.any(interior):
        stride = ridge_max + 1
        encoded = wide_labels.astype(np.int64) * stride + ridge_labels.astype(np.int64)
        unique_pairs = np.unique(encoded[interior])
        wide_ids = (unique_pairs // stride).astype(np.int32)
        ridge_hits = np.bincount(wide_ids, minlength=wide_max + 1).astype(np.int32)
        if not separator_aware:
            keep_wide = ridge_hits <= 1
            keep_wide[0] = False

    kept_multi = 0
    dropped_sep = 0
    if separator_aware and evidence is not None:
        band_evidence = evidence_from_consolidation(evidence)
        for wide_id in range(1, wide_max + 1):
            if ridge_hits[wide_id] <= 1:
                continue
            region = wide_labels == wide_id
            if region_has_internal_separator(region, band_evidence):
                keep_wide[wide_id] = False
                dropped_sep += 1
            else:
                kept_multi += 1

    combined = np.zeros(wide_labels.shape, dtype=np.int32)
    wide_ids = np.unique(wide_labels)
    wide_ids = wide_ids[wide_ids > 0]
    wide_map = np.zeros(wide_max + 1, dtype=np.int32)
    next_id = 1
    for wide_id in wide_ids:
        if not keep_wide[int(wide_id)]:
            continue
        wide_map[int(wide_id)] = next_id
        next_id += 1
    kept_pixels = keep_wide[np.clip(wide_labels, 0, wide_max)]
    combined[kept_pixels] = wide_map[wide_labels[kept_pixels]]

    blocked = cv2.dilate((combined > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    remaining = ridge_labels.copy()
    remaining[blocked > 0] = 0
    ridge_ids = np.unique(remaining)
    ridge_ids = ridge_ids[ridge_ids > 0]
    ridge_map = np.zeros(ridge_max + 1, dtype=np.int32)
    for ridge_id in ridge_ids:
        ridge_map[int(ridge_id)] = next_id
        next_id += 1
    combined[remaining > 0] = ridge_map[remaining[remaining > 0]]
    return combined, kept_multi, dropped_sep


def _connected_with_stats(
    mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    binary = ensure_binary_mask(mask)
    empty_stats = np.zeros((1, 5), dtype=np.int32)
    empty_centroids = np.zeros((1, 2), dtype=np.float64)
    if not np.any(binary):
        return 0, np.zeros(binary.shape, dtype=np.int32), empty_stats, empty_centroids
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return int(count - 1), labels.astype(np.int32), stats, centroids


def _positive_id_count(labels: np.ndarray) -> int:
    if labels.size == 0 or not np.any(labels):
        return 0
    ids = np.unique(labels)
    return int(np.count_nonzero(ids > 0))


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros(labels.shape, dtype=np.int32)
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    lookup[ids] = np.arange(1, int(ids.size) + 1, dtype=np.int32)
    out = np.zeros(labels.shape, dtype=np.int32)
    positive = labels > 0
    out[positive] = lookup[labels[positive]]
    return out


def _empty_band_images(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    empty = np.zeros((*shape, 3), dtype=np.uint8)
    return {
        "metal_structural_conductor_bands": empty,
        "metal_structural_transverse_samples": empty,
        "metal_structural_band_groups_accepted": empty,
        "metal_structural_band_groups_rejected": empty,
    }


def _find(parent: np.ndarray, item: int) -> int:
    root = item
    while parent[root] != root:
        root = int(parent[root])
    while parent[item] != item:
        nxt = int(parent[item])
        parent[item] = root
        item = nxt
    return int(root)


def _union(parent: np.ndarray, left: int, right: int) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _remap_by_parent(labels: np.ndarray, parent: np.ndarray) -> np.ndarray:
    roots = np.arange(parent.size, dtype=np.int32)
    for index in range(1, parent.size):
        roots[index] = _find(parent, index)
    remapped = np.zeros(labels.shape, dtype=np.int32)
    positive = labels > 0
    remapped[positive] = roots[labels[positive]]
    return _compact_labels(remapped)


def _acute_angle_delta(first: float, second: float) -> float:
    delta = abs((first - second + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
    return float(delta)


def _label_overlay(labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = labels > 0
    if not np.any(positive):
        return color
    hsv = np.zeros((*labels.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = ((labels.astype(np.int32) * 37) % 180).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.where(positive, 220, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _draw_links(
    base: np.ndarray,
    links: tuple[RidgeLink, ...],
    color: tuple[int, int, int],
    *,
    reason: str | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(ensure_uint8(base), cv2.COLOR_GRAY2BGR)
    canvas = (canvas.astype(np.float32) * 0.55).astype(np.uint8)
    for link in links:
        if reason is not None and link.reason != reason:
            continue
        cv2.line(canvas, link.point_a, link.point_b, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, link.point_a, 1, color, -1)
        cv2.circle(canvas, link.point_b, 1, color, -1)
    return canvas


def _render_debug(
    intensity: np.ndarray,
    *,
    ridge_raw: np.ndarray,
    ridge_logical: np.ndarray,
    wide_raw: np.ndarray,
    wide_logical: np.ndarray,
    combined: np.ndarray,
    accepted: tuple[RidgeLink, ...],
    rejected: tuple[RidgeLink, ...],
) -> dict[str, np.ndarray]:
    return {
        "metal_structural_ridge_fragments": _label_overlay(ridge_raw),
        "metal_structural_wide_fragments": _label_overlay(wide_raw),
        "metal_structural_ridge_links_accepted": _draw_links(
            intensity, accepted, (0, 220, 0)
        ),
        "metal_structural_ridge_links_rejected": _draw_links(
            intensity, rejected, (0, 200, 255)
        ),
        "metal_structural_ridge_links_boundary_veto": _draw_links(
            intensity,
            rejected,
            (0, 0, 255),
            reason="boundary_veto",
        ),
        "metal_structural_logical_ridge": _label_overlay(ridge_logical),
        "metal_structural_logical_wide": _label_overlay(wide_logical),
        "metal_structural_logical_markers": _label_overlay(combined),
    }
