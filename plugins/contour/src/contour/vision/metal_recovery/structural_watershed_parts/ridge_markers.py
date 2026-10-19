"""structural_watershed: ridge markers."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np

from contour.vision.metal_recovery.structural_watershed_parts.module_state import (
    _NMS_DIRECTIONS,
    _RIBBON_HALF_WIDTHS,
)

from contour.vision.metal_recovery.structural_watershed_parts.extract_structural_features import _angles_aligned

def _ridge_markers(
    features: _StructuralFeatures,
    config: StructuralWatershedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    ribbon = _paired_rim_center_response(features)
    along = features.structure_orientation + (np.pi * 0.5)
    ribbon_nms = _non_maximum_suppress(ribbon, along)
    ribbon_peak = float(np.percentile(ribbon, 99.5)) if ribbon.size else 0.0
    coherent = features.coherence >= float(config.min_orientation_coherence)
    ribbon_seeds = np.where(
        (ribbon_nms > 0)
        & (ribbon >= float(config.min_ridge_confidence) * max(ribbon_peak, 1e-6))
        & coherent,
        255,
        0,
    ).astype(np.uint8)

    hessian_nms = _non_maximum_suppress(features.ridge_response, features.ridge_orientation)
    along_structure = features.structure_orientation + (np.pi * 0.5)
    aligned = _angles_aligned(
        features.ridge_orientation,
        along_structure,
        max_delta=np.pi / 5.0,
    )
    thin_line = ribbon < 0.25 * max(ribbon_peak, 1e-6)
    hessian_seeds = np.where(
        (hessian_nms > 0)
        & (features.ridge_confidence >= float(config.min_ridge_confidence))
        & coherent
        & aligned
        & thin_line,
        255,
        0,
    ).astype(np.uint8)
    raw = cv2.bitwise_or(ribbon_seeds, hessian_seeds)
    transverse_block = features.persistent_edge >= float(np.percentile(features.persistent_edge, 75.0))
    linked = _link_along_orientation(
        raw,
        along,
        length_px=int(config.gap_link_px),
        block=transverse_block,
    )
    return raw, _filter_short_components(
        linked,
        min_area=int(config.min_marker_area),
        min_length=int(config.min_marker_length),
    )

def _paired_rim_center_response(features: _StructuralFeatures) -> np.ndarray:
    """Scale-selected midline between a pair of bright rims with a bright fill."""

    from contour.vision.metal_recovery.structural_watershed_parts.geodesic_label_competition import _unit_range
    across_x = np.cos(features.structure_orientation).astype(np.float32)
    across_y = np.sin(features.structure_orientation).astype(np.float32)
    rim = _unit_range(features.rim_response)
    intensity = features.denoised.astype(np.float32)
    height, width = intensity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    best = np.zeros(intensity.shape, dtype=np.float32)
    for half in _RIBBON_HALF_WIDTHS:
        left_rim = _remap_offset(rim, grid_x, grid_y, across_x * half, across_y * half)
        right_rim = _remap_offset(rim, grid_x, grid_y, -across_x * half, -across_y * half)
        outer_left = _remap_offset(
            intensity,
            grid_x,
            grid_y,
            across_x * (half + 3.0),
            across_y * (half + 3.0),
        )
        outer_right = _remap_offset(
            intensity,
            grid_x,
            grid_y,
            -across_x * (half + 3.0),
            -across_y * (half + 3.0),
        )
        pair = np.minimum(left_rim, right_rim)
        brighter_than_outside = np.minimum(intensity - outer_left, intensity - outer_right)
        score = pair * np.maximum(brighter_than_outside, 0.0) * (1.0 - rim)
        best = np.maximum(best, score)
    return best.astype(np.float32)

def _remap_offset(
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    return cv2.remap(
        values.astype(np.float32),
        grid_x + dx,
        grid_y + dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

def _non_maximum_suppress(response: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Keep local maxima of ridge response in the direction across the ridge."""

    across = orientation + (np.pi * 0.5)
    discrete = np.mod(np.round(across / (np.pi / 4.0)).astype(np.int32), 4)
    kept = np.zeros(response.shape, dtype=np.uint8)
    for index, (dx, dy) in enumerate(_NMS_DIRECTIONS):
        direction = discrete == index
        if not np.any(direction):
            continue
        neighbor_a = _shift_no_wrap(response, dx, dy)
        neighbor_b = _shift_no_wrap(response, -dx, -dy)
        local_max = (response >= neighbor_a) & (response >= neighbor_b) & (response > 0)
        kept[direction & local_max] = 255
    return kept

def _shift_no_wrap(values: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    src_y0 = max(0, -dy)
    src_y1 = values.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = values.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = values.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = values.shape[1] - max(0, -dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return shifted
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return shifted

def _link_along_orientation(
    binary: np.ndarray,
    orientation: np.ndarray,
    *,
    length_px: int,
    block: np.ndarray | None = None,
) -> np.ndarray:
    """Close only short longitudinal gaps; never across a transverse boundary."""

    if length_px <= 0 or not np.any(binary):
        return binary
    linked = binary.copy()
    bins = 8
    blocked = None if block is None else block.astype(bool)
    for index in range(bins):
        angle = np.pi * index / bins
        selected = binary.copy()
        delta = np.abs(np.mod(orientation - angle + np.pi * 0.5, np.pi) - np.pi * 0.5)
        selected[delta > (np.pi / bins)] = 0
        if not np.any(selected):
            continue
        kernel = _oriented_line_kernel(2 * length_px + 1, angle)
        closed = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
        grown = (closed > 0) & (binary == 0)
        if blocked is not None:
            closed[grown & blocked] = 0
        linked = cv2.bitwise_or(linked, closed)
    return linked

def _oriented_line_kernel(size: int, angle: float) -> np.ndarray:
    length = max(3, int(size) | 1)
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    dx = float(np.cos(angle))
    dy = float(np.sin(angle))
    for offset in range(-center, center + 1):
        x = round(center + offset * dx)
        y = round(center + offset * dy)
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    kernel[center, center] = 1
    return kernel

def _filter_short_components(mask: np.ndarray, *, min_area: int, min_length: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(mask.shape, dtype=np.uint8)
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(stats.shape[0], dtype=np.uint8)
    for index in range(1, stats.shape[0]):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area >= min_area and max(width, height) >= min_length:
            keep[index] = 1
    return np.where(keep[labels] > 0, 255, 0).astype(np.uint8)

if TYPE_CHECKING:
    from contour.vision.metal_recovery.structural_watershed_parts.clamped_structural_watershed_config import StructuralWatershedConfig
    from contour.vision.metal_recovery.structural_watershed_parts.extract_structural_features import _StructuralFeatures
