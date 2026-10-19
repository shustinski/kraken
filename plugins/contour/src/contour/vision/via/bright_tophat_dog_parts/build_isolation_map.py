"""bright_tophat_dog: build isolation map."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np
from contour.vision.io_normalize import to_gray_u8

from contour.vision.via.bright_tophat_dog_parts.draw_overlay import _as_gray_f32

def _bright_local_peak_mask(gray: np.ndarray, tile_size: int, percentile: float) -> np.ndarray:
    """Local maxima on raw gray above a tiled brightness floor (catches vias on bright metal)."""
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    floor = _tiled_percentile_mask(gray, min(99.5, float(percentile) + 0.5), tile_size)
    work = gray.astype(np.float32)
    work[floor == 0] = 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(work, kernel)
    peaks = ((work >= dil) & (work > 0) & (floor > 0)).astype(np.uint8) * 255
    return peaks

def _tiled_percentile_mask(response: np.ndarray, percentile: float, tile_size: int) -> np.ndarray:
    if response.size == 0:
        return np.zeros_like(response, dtype=np.uint8)
    tile = max(32, int(tile_size))
    height, width = response.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for y0 in range(0, height, tile):
        y1 = min(height, y0 + tile)
        for x0 in range(0, width, tile):
            x1 = min(width, x0 + tile)
            tile_data = response[y0:y1, x0:x1]
            if tile_data.size == 0:
                continue
            threshold = float(np.percentile(tile_data, float(percentile)))
            mask[y0:y1, x0:x1] = np.where((tile_data >= threshold) & (tile_data > 0), 255, 0).astype(np.uint8)
    return mask

def _odd_kernel_size(value: int) -> int:
    size = max(3, int(value))
    if size % 2 == 0:
        size += 1
    return size

def _build_isolation_map(gray: np.ndarray, max_radius: int) -> tuple[np.ndarray, np.ndarray] | None:
    if gray.size == 0:
        return None
    _threshold, otsu = cv2.threshold(to_gray_u8(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_area = float(cv2.countNonZero(otsu))
    if mask_area <= 0.0:
        return None
    if mask_area / float(otsu.size) > 0.55:
        mask = otsu
    else:
        threshold = float(np.percentile(gray, 92.0))
        mask = np.where((gray >= threshold) & (gray > 0), 255, 0).astype(np.uint8)
        # On flat / binary frames the 92nd percentile collapses to (near) zero and
        # the percentile mask floods, merging every via into one giant component
        # (which zeroes out all isolation scores). Detect that and fall back to the
        # sparse Otsu mask so each via stays its own component.
        if float(cv2.countNonZero(mask)) / float(mask.size) > 0.40:
            mask = otsu
    kernel_size = _odd_kernel_size(max(3, round(max_radius * 0.45)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(mask) <= 0:
        return None
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    return labels, stats

def _isolation_score(
    isolation: tuple[np.ndarray, np.ndarray] | None,
    cx: int,
    cy: int,
    radius: int,
) -> float:
    if isolation is None:
        return 1.0
    labels, stats = isolation
    if cy < 0 or cx < 0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
        return 0.35
    label = int(labels[cy, cx])
    if label <= 0 or label >= stats.shape[0]:
        return 0.35
    width = float(stats[label, cv2.CC_STAT_WIDTH])
    height = float(stats[label, cv2.CC_STAT_HEIGHT])
    diameter = float(radius * 2 + 1)
    elongation = max(width, height) / max(1.0, min(width, height))
    span = max(width, height)
    size_score = 1.0 if span <= diameter * 2.5 else max(0.0, 1.0 - (span - diameter * 2.5) / (diameter * 3.0))
    elongation_score = 1.0 if elongation <= 3.0 else max(0.0, 1.0 - (elongation - 3.0) / 5.0)
    return min(size_score, elongation_score)

def _nms_quality(det: BrightViaDetection) -> float:
    return float(det.circularity) * max(1.0, float(det.annular_contrast))

def _raw_nms_rank(raw: _RawStage1) -> float:
    # Via-specific responses take precedence over generic gray-level peaks when
    # two candidates describe the same location. This is a quality ordering,
    # independent of how many candidates exist in the image.
    source_priority = {
        "matched_peak": 3.0,
        "contour": 2.0,
        "tophat_peak": 1.0,
        "dog_peak": 1.0,
    }.get(raw.source, 0.0)
    quality = float(raw.prelim) * max(0.2, min(1.0, float(raw.circularity) * 1.2))
    return source_priority * 256.0 + quality

def _metal_mask(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    _t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def _edge_distance_map(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    blur = cv2.GaussianBlur(to_gray_u8(gray), (3, 3), 0.0)
    median = float(np.median(blur))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(blur, lower, upper)
    return cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)

def _expanded_bbox(center: tuple[float, float], size: float, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    half = max(1, round(float(size) * 0.5))
    cx, cy = round(center[0]), round(center[1])
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(int(shape[1]), cx + half + 1)
    bottom = min(int(shape[0]), cy + half + 1)
    return left, top, max(0, right - left), max(0, bottom - top)

def _disk_mean(image: np.ndarray, center: tuple[float, float], radius: float) -> float:
    data = _as_gray_f32(image)
    return _disk_mean_f32(data, center, radius)

def _disk_mean_f32(data: np.ndarray, center: tuple[float, float], radius: float) -> float:
    if data.size == 0:
        return 0.0
    bbox = _expanded_bbox(center, float(radius) * 2.0, data.shape)
    x_coord, y_coord, width, height = bbox
    if width <= 0 or height <= 0:
        return 0.0
    patch = data[y_coord : y_coord + height, x_coord : x_coord + width]
    yy = (np.arange(y_coord, y_coord + height, dtype=np.float32) - float(center[1])).reshape(-1, 1)
    xx = (np.arange(x_coord, x_coord + width, dtype=np.float32) - float(center[0])).reshape(1, -1)
    disk = xx * xx + yy * yy <= max(1.0, float(radius)) ** 2
    if not np.any(disk):
        return 0.0
    return float(patch[disk].mean())

def _local_patch(data: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    if data.size == 0:
        return data
    pad = max(1, round(float(radius)))
    left = max(0, round(cx) - pad)
    right = min(data.shape[1], round(cx) + pad + 1)
    top = max(0, round(cy) - pad)
    bottom = min(data.shape[0], round(cy) + pad + 1)
    if right <= left or bottom <= top:
        return np.zeros((0, 0), dtype=data.dtype)
    return data[top:bottom, left:right]

def _bilinear_sample(data: np.ndarray, x_coord: float, y_coord: float) -> float:
    if data.size == 0:
        return 0.0
    x_max = max(0.0, float(data.shape[1] - 1.0))
    y_max = max(0.0, float(data.shape[0] - 1.0))
    if x_coord < 0.0:
        x_coord = 0.0
    elif x_coord > x_max:
        x_coord = x_max
    if y_coord < 0.0:
        y_coord = 0.0
    elif y_coord > y_max:
        y_coord = y_max
    x0 = int(np.floor(x_coord))
    y0 = int(np.floor(y_coord))
    x1 = min(data.shape[1] - 1, x0 + 1)
    y1 = min(data.shape[0] - 1, y0 + 1)
    dx = x_coord - x0
    dy = y_coord - y0
    top = float(data[y0, x0]) * (1.0 - dx) + float(data[y0, x1]) * dx
    bottom = float(data[y1, x0]) * (1.0 - dx) + float(data[y1, x1]) * dx
    return float(top * (1.0 - dy) + bottom * dy)

def _sample_is_inside(data: np.ndarray, x_coord: float, y_coord: float) -> bool:
    return bool(
        data.size
        and 0.0 <= float(x_coord) <= float(data.shape[1] - 1)
        and 0.0 <= float(y_coord) <= float(data.shape[0] - 1)
    )

def _sample_distance(distance_map: np.ndarray, center: tuple[float, float]) -> float:
    if distance_map.size == 0:
        return 0.0
    return _bilinear_sample(distance_map, center[0], center[1])

if TYPE_CHECKING:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetection, _RawStage1
