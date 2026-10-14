"""repository: retain core supported region."""
from __future__ import annotations
import numpy as np

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.candidate_border_span_features import _compact_core_mask

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.label_components import (
    _binary_dilate_rect,
    _label_components,
)

def _retain_core_supported_region(
    probability: np.ndarray,
    prominence_map: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    core_threshold: float,
    grow_threshold: float,
    prominence_threshold: float,
    max_core_aspect_ratio: float,
) -> np.ndarray:
    """Trim a spill-like candidate to regions supported by compact high-prominence cores."""

    from neuralimage.Validation_gradient_widget_lite.core.repository_parts.mask_geometry import _mask_bbox
    mask_bool = np.asarray(candidate_mask, dtype=bool)
    compact_core = _compact_core_mask(
        probability,
        mask_bool,
        core_threshold=core_threshold,
        max_aspect_ratio=max_core_aspect_ratio,
    )
    if not np.any(compact_core):
        return np.zeros_like(mask_bool, dtype=bool)
    support_mask = mask_bool & (
        (np.asarray(probability, dtype=np.float32) >= float(grow_threshold))
        | (np.asarray(prominence_map, dtype=np.float32) >= float(prominence_threshold))
    )
    if not np.any(support_mask):
        return compact_core
    core_labels, core_count = _label_components(compact_core)
    retained = np.zeros_like(mask_bool, dtype=bool)
    for core_id in range(1, int(core_count) + 1):
        core_component = core_labels == core_id
        if not np.any(core_component):
            continue
        core_bbox = _mask_bbox(core_component)
        core_width = max(1, int(core_bbox[2]))
        core_height = max(1, int(core_bbox[3]))
        if core_width >= core_height:
            core_window = _binary_dilate_rect(
                core_component,
                radius_y=max(1, min(4, core_height)),
                radius_x=max(2, min(4, core_width // 3 + 1)),
            )
        else:
            core_window = _binary_dilate_rect(
                core_component,
                radius_y=max(2, min(4, core_height // 3 + 1)),
                radius_x=max(1, min(4, core_width)),
            )
        local_support = support_mask & core_window
        if not np.any(local_support):
            retained |= core_component
            continue
        labels, count = _label_components(local_support)
        retained_core = False
        for label_id in range(1, int(count) + 1):
            component_mask = labels == label_id
            if np.any(component_mask & core_component):
                retained |= component_mask
                retained_core = True
        if not retained_core:
            retained |= core_component
    return retained if np.any(retained) else compact_core
