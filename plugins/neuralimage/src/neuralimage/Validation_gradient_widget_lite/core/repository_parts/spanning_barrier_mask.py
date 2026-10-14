"""repository: spanning barrier mask."""
from __future__ import annotations
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _label_components

def _spanning_barrier_mask(
    score_map: np.ndarray,
    band_mask: np.ndarray,
    *,
    threshold: float,
    horizontal: bool,
    coverage_min: float,
) -> np.ndarray:
    """Find low-score separator components that span the minor axis inside a local band."""

    band_bool = np.asarray(band_mask, dtype=bool)
    if not np.any(band_bool):
        return np.zeros_like(band_bool, dtype=bool)
    valley_mask = band_bool & (np.asarray(score_map, dtype=np.float32) <= float(threshold))
    if not np.any(valley_mask):
        return np.zeros_like(band_bool, dtype=bool)
    labels, count = _label_components(valley_mask)
    if count <= 0:
        return np.zeros_like(band_bool, dtype=bool)
    active_minor = np.any(band_bool, axis=1 if horizontal else 0)
    minor_span = max(1, int(np.count_nonzero(active_minor)))
    spanning = np.zeros_like(band_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        if not np.any(component):
            continue
        ys, xs = np.nonzero(component)
        if ys.size == 0 or xs.size == 0:
            continue
        component_minor_span = int(np.max(ys) - np.min(ys) + 1) if horizontal else int(np.max(xs) - np.min(xs) + 1)
        if float(component_minor_span / max(1, minor_span)) >= float(coverage_min):
            spanning |= component
    return spanning
