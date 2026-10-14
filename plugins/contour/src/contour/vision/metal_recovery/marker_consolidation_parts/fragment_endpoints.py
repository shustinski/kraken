"""marker_consolidation: fragment endpoints."""
from __future__ import annotations
import numpy as np

from contour.vision.metal_recovery.marker_consolidation_parts.module_state import _SEARCH_RADIUS_PX

from contour.vision.metal_recovery.marker_consolidation_parts.continuation_link import _continuation_link

from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import _Endpoint

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
