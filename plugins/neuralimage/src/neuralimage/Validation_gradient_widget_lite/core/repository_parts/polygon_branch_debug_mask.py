"""repository: polygon branch debug mask."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

def _polygon_branch_debug_mask(shape: tuple[int, int], candidates: tuple[_PolygonConfidenceCandidate, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        mask |= np.asarray(candidate.mask, dtype=bool)
    return mask

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.polygon_confidence_config import _PolygonConfidenceCandidate
