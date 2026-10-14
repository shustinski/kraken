"""structural_watershed: wide interior markers."""
from __future__ import annotations
import cv2
import numpy as np
from contour.utils import ensure_binary_mask

from contour.vision.metal_recovery.structural_watershed_parts.ridge_markers import (
    _filter_short_components,
    _shift_no_wrap,
)

def _wide_interior_markers(
    features: _StructuralFeatures,
    existing_core_seeds: np.ndarray,
    config: StructuralWatershedConfig,
) -> np.ndarray:
    """Confident interiors of wide conductors, including dark rim-bounded fills."""

    radius = max(1.0, float(config.wide_interior_radius))
    from_cores = _eroded_wide_cores(existing_core_seeds, radius)
    rim_walls = _rim_walls(features)
    open_space = np.where(rim_walls > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    if count <= 1:
        return from_cores
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    image_area = float(open_space.size)
    contact = _label_rim_contact(labels, count, rim_walls)
    gradient_means = _label_mean(labels, count, features.magnitude)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    box_perimeter = np.maximum(2.0 * (widths + heights), 1.0)
    gradient_limit = max(float(np.percentile(features.magnitude, 60.0)), 8.0)
    keep = (
        (areas >= max(float(config.min_marker_area), 16.0 * radius * radius))
        & (areas <= 0.45 * image_area)
        & (contact >= 0.55 * box_perimeter)
        & (np.minimum(widths, heights) >= 2.0 * radius)
        & (gradient_means <= gradient_limit)
    )
    keep[0] = False
    enclosed = np.where(keep[labels], 255, 0).astype(np.uint8)
    enclosed = _erode_binary(enclosed, 1)
    enclosed = _filter_short_components(
        enclosed,
        min_area=int(max(config.min_marker_area, radius * radius)),
        min_length=int(config.min_marker_length),
    )
    return cv2.bitwise_or(from_cores, enclosed)

def _eroded_wide_cores(core_seeds: np.ndarray, radius: float) -> np.ndarray:
    cores = ensure_binary_mask(core_seeds)
    if not np.any(cores):
        return cores
    distance = cv2.distanceTransform((cores > 0).astype(np.uint8), cv2.DIST_L2, 3)
    wide = np.where(distance >= radius, 255, 0).astype(np.uint8)
    if np.any(wide):
        return wide
    return _erode_binary(cores, max(1, round(radius)))

def _rim_walls(features: _StructuralFeatures) -> np.ndarray:
    rim_limit = float(np.percentile(features.rim_response, 85.0))
    walls = np.where(features.rim_response >= max(rim_limit, 1e-4), 255, 0).astype(np.uint8)
    return cv2.dilate(walls, np.ones((5, 5), np.uint8))

def _label_rim_contact(labels: np.ndarray, count: int, rim_walls: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(rim_walls, np.ones((3, 3), np.uint8))
    return np.bincount(
        labels.ravel(),
        weights=(dilated > 0).ravel().astype(np.float64),
        minlength=count,
    )

def _label_mean(labels: np.ndarray, count: int, values: np.ndarray) -> np.ndarray:
    sums = np.bincount(
        labels.ravel(),
        weights=values.astype(np.float64).ravel(),
        minlength=count,
    )
    counts = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
    return np.divide(sums, np.maximum(counts, 1.0))

def _erode_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return ensure_binary_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(ensure_binary_mask(mask), kernel)

def _combine_foreground_markers(ridge_markers: np.ndarray, wide_markers: np.ndarray) -> np.ndarray:
    """Keep simple wide interiors; replace merged interiors with their ridges.

    A wide core that already covers several parallel traces would recreate the
    merge failure. Those cores are dropped and the interior ridge centerlines
    become the instance markers. Ridges that only touch a kept plate are rims
    and are discarded so they cannot split it.
    """

    ridges = ensure_binary_mask(ridge_markers)
    wide = ensure_binary_mask(wide_markers)
    if not np.any(ridges):
        return wide
    if not np.any(wide):
        return ridges

    ridge_count, ridge_labels = cv2.connectedComponents((ridges > 0).astype(np.uint8), connectivity=8)
    wide_count, wide_labels = cv2.connectedComponents((wide > 0).astype(np.uint8), connectivity=8)
    interior = (ridge_labels > 0) & (wide_labels > 0)
    interior_ridge_count = np.zeros(wide_count, dtype=np.int32)
    if np.any(interior):
        pair = wide_labels.astype(np.int64) * int(ridge_count) + ridge_labels.astype(np.int64)
        unique_pairs = np.unique(pair[interior])
        wide_ids = (unique_pairs // int(ridge_count)).astype(np.int32)
        interior_ridge_count = np.bincount(wide_ids, minlength=wide_count).astype(np.int32)

    keep_wide = interior_ridge_count <= 1
    keep_wide[0] = False
    kept_wide = np.where(keep_wide[wide_labels], 255, 0).astype(np.uint8)
    kept_wide = _fill_small_holes(kept_wide, max_area=64)

    blocked = cv2.dilate((kept_wide > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    kept_ridges = ridges.copy()
    kept_ridges[blocked > 0] = 0
    return cv2.bitwise_or(kept_wide, kept_ridges)

def _fill_small_holes(mask: np.ndarray, *, max_area: int) -> np.ndarray:
    binary = ensure_binary_mask(mask)
    if not np.any(binary):
        return binary
    inverted = np.where(binary > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(inverted, connectivity=4)
    filled = binary.copy()
    height, width = binary.shape
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        right = left + int(stats[index, cv2.CC_STAT_WIDTH])
        bottom = top + int(stats[index, cv2.CC_STAT_HEIGHT])
        touches_border = left <= 0 or top <= 0 or right >= width or bottom >= height
        if (not touches_border) and area <= max_area:
            filled[labels == index] = 255
    return filled

def _background_markers(
    features: _StructuralFeatures,
    groove_seeds: np.ndarray,
    ridge_markers: np.ndarray,
    foreground_markers: np.ndarray,
    config: StructuralWatershedConfig,
) -> np.ndarray:
    """Background only in high-confidence separators, never inside traces.

    Existing groove seeds are built for a binary core/groove watershed and are
    too dense for instance markers: they occupy conductor interiors and turn
    merges into misses. Inter-ridge valleys handle narrow gaps; distant dark
    substrate handles open fields.
    """

    del groove_seeds
    valleys = _inter_ridge_valleys(features, ridge_markers, config)
    substrate = _confident_substrate(features, foreground_markers)
    background = cv2.bitwise_or(valleys, substrate)
    blocked = cv2.dilate(ensure_binary_mask(foreground_markers), np.ones((3, 3), np.uint8))
    background[blocked > 0] = 0
    return ensure_binary_mask(background)

def _confident_substrate(features: _StructuralFeatures, foreground_markers: np.ndarray) -> np.ndarray:
    """High-confidence substrate: dark, calm, and far from foreground markers."""

    intensity = features.denoised.astype(np.float32)
    dark = intensity <= float(np.percentile(intensity, 30.0))
    calm = features.magnitude <= float(np.percentile(features.magnitude, 35.0))
    far = cv2.dilate(ensure_binary_mask(foreground_markers), np.ones((21, 21), np.uint8)) == 0
    seeds = np.where(dark & calm & far, 255, 0).astype(np.uint8)
    return _filter_short_components(seeds, min_area=32, min_length=8)

def _inter_ridge_valleys(
    features: _StructuralFeatures,
    ridge_markers: np.ndarray,
    config: StructuralWatershedConfig,
) -> np.ndarray:
    ridges = ensure_binary_mask(ridge_markers)
    if not np.any(ridges):
        return np.zeros(ridges.shape, dtype=np.uint8)
    distance = cv2.distanceTransform(cv2.bitwise_not(ridges), cv2.DIST_L2, 3)
    local_max = (
        (distance >= _shift_no_wrap(distance, 1, 0))
        & (distance >= _shift_no_wrap(distance, -1, 0))
        & (distance >= _shift_no_wrap(distance, 0, 1))
        & (distance >= _shift_no_wrap(distance, 0, -1))
        & (distance >= 1.0)
        & (distance <= max(2.0, float(config.wide_interior_radius) * 2.0))
    )
    low_ridge = features.ridge_confidence <= float(config.min_ridge_confidence)
    dark = features.denoised.astype(np.float32) <= float(np.percentile(features.denoised, 30.0))
    valleys = np.where(local_max & low_ridge & dark, 255, 0).astype(np.uint8)
    return _filter_short_components(
        valleys,
        min_area=max(2, int(config.min_marker_area) // 2),
        min_length=max(3, int(config.min_marker_length) // 2),
    )
