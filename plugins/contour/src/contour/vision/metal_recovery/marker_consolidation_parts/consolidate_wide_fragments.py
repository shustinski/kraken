"""marker_consolidation: consolidate wide fragments."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from contour.vision.metal_recovery.marker_consolidation_parts.enclosed_conductor_basins import (
    _basin_has_parallel_ridges,
    _enclosed_conductor_basins,
)

from contour.vision.metal_recovery.marker_consolidation_parts.fragment_majority_labels import _fragment_majority_labels

from contour.vision.metal_recovery.marker_consolidation_parts.local_wide_unions import _local_wide_unions

from contour.vision.metal_recovery.marker_consolidation_parts.render_debug import (
    _find,
    _remap_by_parent,
    _union,
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

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence
