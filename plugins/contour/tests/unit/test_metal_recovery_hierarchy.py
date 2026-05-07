from __future__ import annotations

import cv2
import numpy as np

from contour.domain import PolygonData, compute_polygon_metrics
from contour.vision.metal_recovery.detector import (
    MetalRecoveryConfig,
    _append_hierarchy_descendants,
)


def test_metal_recovery_retr_tree_adds_nested_inner_conductor() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (74, 74), 255, -1)
    cv2.rectangle(mask, (20, 20), (60, 60), 0, -1)
    cv2.rectangle(mask, (34, 34), (46, 46), 255, -1)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert hierarchy is not None
    hierarchy_array = hierarchy[0]
    root_index = next(idx for idx, row in enumerate(hierarchy_array) if int(row[3]) == -1)

    root = PolygonData(
        id=1,
        points=[(5.0, 5.0), (74.0, 5.0), (74.0, 74.0), (5.0, 74.0)],
        is_hole=False,
        parent_id=None,
        category="conductor",
        shape_hint="polygon",
    )
    root.area, root.perimeter, root.bbox = compute_polygon_metrics(root.points)
    accepted = [root]
    accepted_mask = np.zeros_like(mask)
    cv2.drawContours(accepted_mask, [contours[root_index]], 0, 255, thickness=-1)

    added = _append_hierarchy_descendants(
        accepted,
        accepted_mask,
        contours,
        hierarchy_array,
        {root_index: root.id},
        MetalRecoveryConfig(
            min_area=10.0,
            min_perimeter=8.0,
            min_width_px=1.0,
            min_inner_hole_area=10.0,
            min_points=3,
            min_polygon_angle_deg=0.0,
        ),
    )

    assert added == 2
    assert len(accepted) == 3
    assert any(polygon.is_hole for polygon in accepted)
    assert any(not polygon.is_hole and polygon.parent_id is not None for polygon in accepted)
