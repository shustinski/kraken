"""Build hierarchical polygon list from a filled (segmentation) mask."""

from __future__ import annotations

import cv2
import numpy as np

from ...domain import integer_points
from ..schemas import HierarchicalComponent

try:
    from ...utils import ensure_uint8
except ImportError:  # pragma: no cover
    from ..io_normalize import ensure_uint8_local as ensure_uint8  # type: ignore[misc,assignment]


def build_hierarchy_from_mask(
    mask: np.ndarray,
    *,
    epsilon: float = 1.0,
) -> tuple[np.ndarray, list[HierarchicalComponent]]:
    """``RETR_TREE`` on binarized mask; returns simplified contours + metadata."""

    m = (ensure_uint8(mask) > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return np.array([], dtype=object), []

    h = hierarchy[0]
    components: list[HierarchicalComponent] = []
    for index, c in enumerate(contours):
        if len(c) < 3:
            continue
        if epsilon > 0.0:
            c = cv2.approxPolyDP(c, float(epsilon), True)
        flat = c.reshape(-1, 2)
        points = integer_points([(float(p[0]), float(p[1])) for p in flat])
        if len(points) < 3:
            continue
        parent_idx = int(h[index][3])
        parent_id = int(parent_idx + 1) if parent_idx >= 0 else None
        depth = 0
        pwalk = index
        while int(h[pwalk][3]) >= 0:
            depth += 1
            pwalk = int(h[pwalk][3])
        is_hole = depth % 2 == 1
        area = float(abs(cv2.contourArea(c)))
        x0, y0, bw, bh = cv2.boundingRect(c)
        comp = HierarchicalComponent(
            id=index + 1,
            contour_index=index,
            parent_id=parent_id,
            depth=depth,
            is_hole=is_hole,
            points=points,
            area=area,
            bbox_xywh=(x0, y0, bw, bh),
        )
        components.append(comp)
    return contours, components
