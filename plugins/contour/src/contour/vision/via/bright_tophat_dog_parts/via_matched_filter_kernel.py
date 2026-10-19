"""bright_tophat_dog: via matched filter kernel."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np
from contour.vision.io_normalize import to_gray_u8

from contour.vision.via.bright_tophat_dog_parts.build_isolation_map import (
    _bilinear_sample,
    _disk_mean_f32,
    _local_patch,
    _nms_quality,
    _sample_is_inside,
)

from contour.vision.via.bright_tophat_dog_parts.center_distance_index import _CenterDistanceIndex

from contour.vision.via.bright_tophat_dog_parts.draw_overlay import _as_gray_f32

def radial_symmetry_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    data = _as_gray_f32(gray)
    return _radial_symmetry_score_f32(data, cx, cy, radius)

def _radial_symmetry_score_f32(data: np.ndarray, cx: float, cy: float, radius: float) -> float:
    if data.size == 0:
        return 0.0
    radius = max(1.0, float(radius))
    offsets = (
        (0.0, -1.0),
        (0.70710678, -0.70710678),
        (1.0, 0.0),
        (0.70710678, 0.70710678),
        (0.0, 1.0),
        (-0.70710678, 0.70710678),
        (-1.0, 0.0),
        (-0.70710678, -0.70710678),
    )
    points = [(cx + radius * dx, cy + radius * dy) for dx, dy in offsets]
    samples = [_bilinear_sample(data, x, y) for x, y in points if _sample_is_inside(data, x, y)]
    if len(samples) < 3:
        return 0.0
    return float(np.std(np.asarray(samples, dtype=np.float32)))

def edge_likeness_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    data = _as_gray_f32(gray)
    return _edge_likeness_score_f32(data, cx, cy, radius)

def _edge_likeness_score_f32(data: np.ndarray, cx: float, cy: float, radius: float) -> float:
    if data.size == 0:
        return 0.0
    radius = max(1.0, float(radius))
    offsets = (
        ((0.0, -1.0), (0.0, 1.0)),
        ((1.0, 0.0), (-1.0, 0.0)),
        ((0.70710678, -0.70710678), (-0.70710678, 0.70710678)),
        ((0.70710678, 0.70710678), (-0.70710678, -0.70710678)),
    )
    differences: list[float] = []
    for first, second in offsets:
        first_x, first_y = cx + radius * first[0], cy + radius * first[1]
        second_x, second_y = cx + radius * second[0], cy + radius * second[1]
        if not (_sample_is_inside(data, first_x, first_y) and _sample_is_inside(data, second_x, second_y)):
            continue
        differences.append(abs(_bilinear_sample(data, first_x, first_y) - _bilinear_sample(data, second_x, second_y)))
    return float(max(differences, default=0.0))

def line_likeness_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    data = _as_gray_f32(gray)
    return _line_likeness_score_f32(data, cx, cy, radius)

def _line_likeness_score_f32(data: np.ndarray, cx: float, cy: float, radius: float) -> float:
    patch = _local_patch(data, cx, cy, max(2.0, float(radius) * 1.8))
    if patch.size <= 4:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    if trace <= 1e-6:
        return 0.0
    delta = float(np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy))
    return float(np.clip(100.0 * delta / (trace + 1e-6), 0.0, 100.0))

def bright_center_score(gray: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    data = _as_gray_f32(gray)
    return _bright_center_score_f32(data, center, diameter)

def _bright_center_score_f32(data: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    if data.size == 0:
        return 0.0
    cx, cy = float(center[0]), float(center[1])
    radius = max(1.0, float(diameter) * 0.5)
    pad = max(2, round(radius * 2.2))
    left = max(0, round(cx) - pad)
    right = min(data.shape[1], round(cx) + pad + 1)
    top = max(0, round(cy) - pad)
    bottom = min(data.shape[0], round(cy) + pad + 1)
    if right <= left or bottom <= top:
        return 0.0
    patch = data[top:bottom, left:right]
    yy = (np.arange(top, bottom, dtype=np.float32) - cy).reshape(-1, 1)
    xx = (np.arange(left, right, dtype=np.float32) - cx).reshape(1, -1)
    dist2 = xx * xx + yy * yy
    center_mask = dist2 <= max(1.0, radius * 0.45) ** 2
    ring_mask = (dist2 >= max(1.0, radius * 0.85) ** 2) & (dist2 <= max(2.0, radius * 1.65) ** 2)
    if not np.any(center_mask) or not np.any(ring_mask):
        return 0.0
    return float(patch[center_mask].mean() - patch[ring_mask].mean())

def center_mean_score(gray: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    data = _as_gray_f32(gray)
    return _center_mean_score_f32(data, center, diameter)

def _center_mean_score_f32(data: np.ndarray, center: tuple[float, float], diameter: float) -> float:
    if data.size == 0:
        return 0.0
    radius = max(1.0, float(diameter) * 0.22)
    return _disk_mean_f32(data, center, radius)

def mask_fraction(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    data = to_gray_u8(mask)
    if data.size == 0:
        return 0.0
    x_coord, y_coord, width, height = bbox
    left = max(0, int(x_coord))
    top = max(0, int(y_coord))
    right = min(data.shape[1], left + max(0, int(width)))
    bottom = min(data.shape[0], top + max(0, int(height)))
    if right <= left or bottom <= top:
        return 0.0
    patch = data[top:bottom, left:right]
    return float(cv2.countNonZero(patch)) / float(patch.size)

def suppress_close_points(detections: list[BrightViaDetection], distance: int) -> list[BrightViaDetection]:
    if distance <= 0 or not detections:
        return sorted(detections, key=lambda item: (item.center[1], item.center[0]))
    kept: list[BrightViaDetection] = []
    min_dist = float(distance)
    idx = _CenterDistanceIndex(min_dist)
    for candidate in sorted(
        detections,
        key=lambda item: (item.isolation_score, _nms_quality(item), item.final_score),
        reverse=True,
    ):
        cx, cy = candidate.center
        if idx.is_close(cx, cy):
            continue
        kept.append(candidate)
        idx.add(cx, cy)
    return sorted(kept, key=lambda item: (item.center[1], item.center[0]))

def _preprocess(gray: np.ndarray, cfg: BrightViaDetectorConfig) -> np.ndarray:
    tile = max(1, int(cfg.clahe_tile_grid_size))
    processed = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=(tile, tile)).apply(gray)
    if cfg.median_blur_kernel > 1:
        processed = cv2.medianBlur(processed, int(cfg.median_blur_kernel))
    return processed

def _white_tophat(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

def _dog_response(gray: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    data = gray.astype(np.float32)
    small = cv2.GaussianBlur(data, (0, 0), float(sigma_small))
    large = cv2.GaussianBlur(data, (0, 0), float(sigma_large))
    return np.maximum(small - large, 0.0)

def _via_matched_filter_kernel(diameter: float) -> np.ndarray:
    """Zero-DC disk-minus-ring kernel tuned to the via diameter.

    Because the kernel sums to ~0, the response measures *local* bright-disk
    contrast against the immediate surround instead of absolute brightness. That
    makes the candidate stage robust on bright backgrounds (e.g. SEM metal
    stripes) where global percentile thresholds on tophat/DoG drown the vias.
    """

    d = max(3.0, float(diameter))
    radius = d * 0.5
    ring_width = max(1.5, radius * 0.7)
    size = int(np.ceil(d + 2.0 * ring_width)) + 1
    if size % 2 == 0:
        size += 1
    center = size // 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    core = (dist <= radius).astype(np.float32)
    ring = ((dist > radius) & (dist <= radius + ring_width)).astype(np.float32)
    core_sum = float(core.sum())
    ring_sum = float(ring.sum())
    if core_sum <= 0.0 or ring_sum <= 0.0:
        return np.zeros((size, size), dtype=np.float32)
    return (core / core_sum - ring / ring_sum).astype(np.float32)

def _via_matched_filter_response(processed: np.ndarray, diameter: float) -> np.ndarray:
    if processed.size == 0:
        return np.zeros_like(processed, dtype=np.float32)
    kernel = _via_matched_filter_kernel(diameter)
    if kernel.size == 0:
        return np.zeros(processed.shape[:2], dtype=np.float32)
    data = processed.astype(np.float32, copy=False)
    response = cv2.filter2D(data, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)
    np.maximum(response, 0.0, out=response)
    return response

def _percentile_mask(response: np.ndarray, percentile: float) -> np.ndarray:
    if response.size == 0:
        return np.zeros_like(response, dtype=np.uint8)
    threshold = float(np.percentile(response, float(percentile)))
    # On flat / binary images most of the response is exactly 0, so the high
    # percentile collapses to 0 and a plain ``>= threshold`` would flood the whole
    # frame. Require a strictly positive response so the background is never a
    # candidate region.
    return np.where((response >= threshold) & (response > 0), 255, 0).astype(np.uint8)

if TYPE_CHECKING:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetection
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detector_config import BrightViaDetectorConfig
