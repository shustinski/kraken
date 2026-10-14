"""repository: polygon branch debug mask."""
from __future__ import annotations
import numpy as np

def _polygon_branch_debug_mask(shape: tuple[int, int], candidates: tuple[_PolygonConfidenceCandidate, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        mask |= np.asarray(candidate.mask, dtype=bool)
    return mask
