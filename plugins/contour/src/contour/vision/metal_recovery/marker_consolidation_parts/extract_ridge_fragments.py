"""marker_consolidation: extract ridge fragments."""
from __future__ import annotations
import cv2
import numpy as np

from contour.vision.metal_recovery.marker_consolidation_parts.module_state import _NEIGHBOR_KERNEL

from contour.vision.metal_recovery.marker_consolidation_parts.fragment_endpoints import _fragment_endpoints

from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import _Fragment

def _extract_ridge_fragments(
    ridge_labels: np.ndarray,
    ridge_count: int,
    stats: np.ndarray,
    centroids: np.ndarray,
    evidence: ConsolidationEvidence,
) -> tuple[_Fragment, ...]:
    binary = (ridge_labels > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(
        binary.astype(np.float32),
        cv2.CV_32F,
        _NEIGHBOR_KERNEL.astype(np.float32),
    )
    fragments: list[_Fragment] = []
    height, width = ridge_labels.shape
    for fragment_id in range(1, ridge_count + 1):
        left = int(stats[fragment_id, cv2.CC_STAT_LEFT])
        top = int(stats[fragment_id, cv2.CC_STAT_TOP])
        box_w = int(stats[fragment_id, cv2.CC_STAT_WIDTH])
        box_h = int(stats[fragment_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[fragment_id, cv2.CC_STAT_AREA])
        y0 = max(0, top - 1)
        x0 = max(0, left - 1)
        y1 = min(height, top + box_h + 1)
        x1 = min(width, left + box_w + 1)
        roi = ridge_labels[y0:y1, x0:x1] == fragment_id
        if not np.any(roi):
            continue
        mean_intensity = float(np.mean(evidence.intensity[y0:y1, x0:x1][roi]))
        mean_ridge = float(np.mean(evidence.ridge_confidence[y0:y1, x0:x1][roi]))
        distance = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 3)
        width_px = max(float(distance.max()), 1.0)
        centroid_row = float(centroids[fragment_id, 1])
        centroid_col = float(centroids[fragment_id, 0])
        endpoints = _fragment_endpoints(
            fragment_id=fragment_id,
            roi=roi,
            origin_row=y0,
            origin_col=x0,
            neighbor_roi=neighbor_count[y0:y1, x0:x1],
            evidence=evidence,
            centroid_row=centroid_row,
            centroid_col=centroid_col,
            width_px=width_px,
            mean_intensity=mean_intensity,
            mean_ridge=mean_ridge,
        )
        fragments.append(
            _Fragment(
                fragment_id=fragment_id,
                area=area,
                centroid_row=centroid_row,
                centroid_col=centroid_col,
                mean_intensity=mean_intensity,
                mean_ridge=mean_ridge,
                width=width_px,
                endpoints=endpoints,
            )
        )
    return tuple(fragments)
