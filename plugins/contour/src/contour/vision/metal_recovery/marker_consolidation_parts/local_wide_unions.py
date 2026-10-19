"""marker_consolidation: local wide unions."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np

from contour.vision.metal_recovery.marker_consolidation_parts.module_state import _LOCAL_WIDE_RADIUS_PX

from contour.vision.metal_recovery.marker_consolidation_parts.render_debug import (
    _find,
    _union,
)

from contour.vision.metal_recovery.marker_consolidation_parts.sample_corridor import _sample_corridor

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

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence
