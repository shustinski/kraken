"""marker_consolidation: enclosed conductor basins."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np

def _enclosed_conductor_basins(
    evidence: ConsolidationEvidence,
    *,
    min_marker_area: int,
    wide_interior_radius: float,
) -> np.ndarray:
    radius = max(1.0, float(wide_interior_radius))
    rim_limit = float(np.percentile(evidence.rim_response, 85.0))
    walls: np.ndarray = np.where(evidence.rim_response >= max(rim_limit, 1e-4), 255, 0).astype(np.uint8)
    walls = cv2.dilate(walls, np.ones((5, 5), np.uint8))
    open_space = np.where(walls > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    if count <= 1:
        return np.zeros(open_space.shape, dtype=np.int32)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    box_perimeter = np.maximum(2.0 * (widths + heights), 1.0)
    dilated = cv2.dilate(walls, np.ones((3, 3), np.uint8))
    contact = np.bincount(
        labels.ravel(),
        weights=(dilated > 0).ravel().astype(np.float64),
        minlength=count,
    )
    gradient_means = np.bincount(
        labels.ravel(),
        weights=evidence.magnitude.astype(np.float64).ravel(),
        minlength=count,
    )
    pixel_counts = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
    gradient_means = np.divide(gradient_means, np.maximum(pixel_counts, 1.0))
    gradient_limit = max(float(np.percentile(evidence.magnitude, 60.0)), 8.0)
    keep = (
        (areas >= max(float(min_marker_area), 16.0 * radius * radius))
        & (areas <= 0.45 * float(open_space.size))
        & (contact >= 0.55 * box_perimeter)
        & (np.minimum(widths, heights) >= 2.0 * radius)
        & (gradient_means <= gradient_limit)
    )
    keep[0] = False
    return np.where(keep[labels], labels, 0).astype(np.int32)

def _basin_has_parallel_ridges(basin: np.ndarray, ridge_labels: np.ndarray) -> bool:
    """True only if two distinct ridges live inside the basin, not merely graze the rim."""

    interior = cv2.erode(basin.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if not np.any(interior) or not np.any(ridge_labels):
        return False
    ridge_max = int(ridge_labels.max())
    total = np.bincount(ridge_labels.ravel(), minlength=ridge_max + 1)
    inside = np.bincount(ridge_labels[interior].ravel(), minlength=ridge_max + 1)
    lives_inside = (inside[1:] * 2 >= total[1:]) & (inside[1:] > 0)
    return int(np.count_nonzero(lives_inside)) >= 2

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence
