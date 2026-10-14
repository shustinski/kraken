"""marker_consolidation: fragment majority labels."""
from __future__ import annotations
import numpy as np

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
