"""repository: assign candidate region seeds."""
from __future__ import annotations
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import _label_components

def _assign_candidate_region_to_seeds(
    candidate_region: np.ndarray,
    seed_labels: np.ndarray,
    blocked_mask: np.ndarray,
) -> np.ndarray:
    """Assign candidate pixels to seed instances while respecting blocked barriers."""
    from neuralimage.Validation_gradient_widget_lite.core.repository import ndi


    candidate_bool = np.asarray(candidate_region, dtype=bool)
    labels = np.asarray(seed_labels, dtype=np.int32)
    blocked = np.asarray(blocked_mask, dtype=bool) & candidate_bool
    if not np.any(candidate_bool) or not np.any(labels > 0):
        return np.zeros_like(labels, dtype=np.int32)
    available = candidate_bool & ~blocked
    available |= labels > 0
    if not np.any(available):
        return np.zeros_like(labels, dtype=np.int32)
    component_labels, component_count = _label_components(available)
    if component_count <= 0:
        return np.zeros_like(labels, dtype=np.int32)
    assigned = np.zeros_like(labels, dtype=np.int32)
    for component_id in range(1, int(component_count) + 1):
        component = component_labels == component_id
        if not np.any(component):
            continue
        touched = np.unique(labels[component])
        touched = touched[touched > 0]
        if touched.size == 1:
            assigned[component] = int(touched[0])
            continue
        if touched.size <= 1:
            continue
        valid_seed_mask = np.isin(labels, touched, assume_unique=True)
        if not np.any(valid_seed_mask):
            continue
        _distances, nearest_indices = ndi.distance_transform_edt(~valid_seed_mask, return_indices=True)
        nearest_seed_labels = labels[tuple(nearest_indices)]
        assigned_component = nearest_seed_labels[component]
        positive = assigned_component > 0
        if np.any(positive):
            assigned_values = assigned_component[positive]
            component_indices = np.argwhere(component)
            assigned[component_indices[positive, 0], component_indices[positive, 1]] = assigned_values
    return assigned
