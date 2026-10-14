"""marker_consolidation: module state."""
from __future__ import annotations
from typing import Literal
import numpy as np

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
