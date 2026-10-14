"""detector: directionally bridge mask."""
from __future__ import annotations
import cv2
import numpy as np
from contour.utils import ensure_uint8

def _directionally_bridge_mask(
    mask: np.ndarray,
    source_gray: np.ndarray,
    *,
    bridge_px: int,
    min_source_intensity: float,
) -> np.ndarray:
    """Bridge horizontal gaps only when the missing source pixels remain bright."""
    radius = max(0, int(bridge_px))
    if radius <= 0 or mask.size == 0:
        return mask.copy()
    selected = ensure_uint8(mask).copy()
    _base_count, base_labels = cv2.connectedComponents(selected, connectivity=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 1))
    closed_all = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
    _closed_count, closed_labels = cv2.connectedComponents(closed_all, connectivity=8)
    required_source = max(0.0, float(min_source_intensity))
    for closed_id in np.unique(closed_labels[closed_labels > 0]):
        region = closed_labels == closed_id
        base_ids = np.unique(base_labels[region & (base_labels > 0)])
        if base_ids.size <= 1:
            continue
        added = region & (selected == 0)
        bridge_added = np.zeros_like(added)
        for row in np.flatnonzero(np.any(added, axis=1)):
            occupied_columns = np.flatnonzero(region[row] & (selected[row] > 0))
            if occupied_columns.size < 2:
                continue
            left = int(occupied_columns.min())
            right = int(occupied_columns.max()) + 1
            bridge_added[row, left:right] = added[row, left:right]
        if bridge_added.any() and float(np.median(source_gray[bridge_added])) >= required_source:
            selected[bridge_added] = 255
    return selected
