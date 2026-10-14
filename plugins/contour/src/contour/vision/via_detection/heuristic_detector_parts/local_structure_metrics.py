"""heuristic_detector: local structure metrics."""
from __future__ import annotations
import math
from functools import lru_cache
import cv2
import numpy as np
from contour.vision.via_detection.config import HeuristicViaDetectorConfig, ViaPolarity

def _refine_center_xy(
    gpatch: np.ndarray,
    mask_bool: np.ndarray,
    *,
    seed_x: float,
    seed_y: float,
    med: float,
    hyp: str,
) -> tuple[float, float]:
    """Centroid with positive contrast weights; falls back to binary moments."""
    ph, pw = gpatch.shape[:2]
    m = mask_bool.astype(np.float32)
    g = gpatch.astype(np.float32)
    if hyp in (str(ViaPolarity.DARK), "dark"):
        wts = (float(med) - g) * m
    elif hyp in (str(ViaPolarity.BRIGHT), "bright"):
        wts = (g - float(med)) * m
    else:
        wts = np.abs(g - float(med)) * m
    wts = np.maximum(wts, 0.0)
    sw = float(wts.sum())
    if sw > 1e-2:
        yy, xx = _coordinate_grids((ph, pw))
        return float((xx * wts).sum() / sw), float((yy * wts).sum() / sw)
    mm = cv2.moments((mask_bool.astype(np.uint8) * 255), binaryImage=True)
    if mm.get("m00", 0) and float(mm["m00"]) > 1e-6:
        return float(mm["m10"] / mm["m00"]), float(mm["m01"] / mm["m00"])
    return float(seed_x), float(seed_y)

def _component_geometry_center(
    contour: np.ndarray,
    stats: np.ndarray,
    weighted_center: tuple[float, float],
) -> tuple[float, float]:
    """Center a via by shape while retaining a little sub-pixel intensity detail."""

    left = float(stats[cv2.CC_STAT_LEFT])
    top = float(stats[cv2.CC_STAT_TOP])
    width = float(stats[cv2.CC_STAT_WIDTH])
    height = float(stats[cv2.CC_STAT_HEIGHT])
    bbox_center = (left + 0.5 * max(0.0, width - 1.0), top + 0.5 * max(0.0, height - 1.0))
    rect_center = tuple(float(value) for value in cv2.minAreaRect(contour)[0])
    circle_center_raw, _radius = cv2.minEnclosingCircle(contour)
    circle_center = tuple(float(value) for value in circle_center_raw)
    geometry_x = float(sorted((bbox_center[0], rect_center[0], circle_center[0]))[1])
    geometry_y = float(sorted((bbox_center[1], rect_center[1], circle_center[1]))[1])
    correction = math.hypot(geometry_x - weighted_center[0], geometry_y - weighted_center[1])
    if correction < 0.5:
        return weighted_center
    return (
        0.70 * geometry_x + 0.30 * float(weighted_center[0]),
        0.70 * geometry_y + 0.30 * float(weighted_center[1]),
    )

@lru_cache(maxsize=512)
def _coordinate_grids(shape: tuple[int, int]) -> np.ndarray:
    return np.indices(shape, dtype=np.float32)

@lru_cache(maxsize=2048)
def _annulus_masks(shape: tuple[int, int], cx: int, cy: int, d: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = float(d) * 0.5
    center = rr <= 0.32 * r
    inner = (rr > 0.34 * r) & (rr <= 0.64 * r)
    outer = (rr > 0.66 * r) & (rr <= 1.08 * r)
    return center, inner, outer

def _mean_mask(patch: np.ndarray, m: np.ndarray) -> float:
    mask_u8 = m.view(np.uint8)
    if cv2.countNonZero(mask_u8) == 0:
        return float(np.mean(patch))
    # OpenCV accumulates directly through the mask. NumPy's patch[m] first
    # allocates a temporary array, which is costly in this per-candidate path.
    return float(cv2.mean(patch, mask=mask_u8)[0])

def _center_in_brightness_range(center_grey: float, hyp: str, config: HeuristicViaDetectorConfig) -> bool:
    if hyp in (str(ViaPolarity.BRIGHT), "bright", str(ViaPolarity.RING_DARK_RING), "ring_dark_ring"):
        if not config.bright_range_enabled:
            return True
        return float(config.bright_range_min) <= center_grey <= float(config.bright_range_max)
    if hyp in (str(ViaPolarity.DARK), "dark", str(ViaPolarity.RING_LIGHT_RING), "ring_light_ring"):
        if not config.dark_range_enabled:
            return True
        return float(config.dark_range_min) <= center_grey <= float(config.dark_range_max)
    return True

def _contrast_for_polarity(
    gray: np.ndarray, cmask: np.ndarray, imask: np.ndarray, omask: np.ndarray, hyp: str
) -> float:
    c = _mean_mask(gray, cmask)
    i = _mean_mask(gray, imask)
    o = _mean_mask(gray, omask)
    if hyp in (str(ViaPolarity.BRIGHT), "bright"):
        return c - o
    if hyp in (str(ViaPolarity.DARK), "dark"):
        return o - c
    if hyp in (str(ViaPolarity.RING_LIGHT_RING), "ring_light_ring"):
        return max(0.0, (i - c) + (i - o))
    if hyp in (str(ViaPolarity.RING_DARK_RING), "ring_dark_ring"):
        return max(0.0, (c - i) + (o - i))
    return 0.0

def _local_center_background_contrast(gray: np.ndarray, cx: int, cy: int, d: float, hyp: str) -> float:
    """Contrast of the via core against a true local background annulus."""

    core, background = _local_background_masks(gray.shape[:2], int(cx), int(cy), float(d))
    core_mean = _mean_mask(gray, core)
    background_mean = _mean_mask(gray, background)
    if hyp in (str(ViaPolarity.DARK), "dark"):
        return background_mean - core_mean
    if hyp in (str(ViaPolarity.BRIGHT), "bright"):
        return core_mean - background_mean
    return 0.0

@lru_cache(maxsize=2048)
def _local_background_masks(
    shape: tuple[int, int], cx: int, cy: int, d: float
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    radius = max(1.0, float(d) * 0.5)
    distance = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    core = distance <= 0.45 * radius
    background = (distance >= 1.25 * radius) & (distance <= 1.85 * radius)
    return core, background

def _local_structure_metrics(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    gradient_magnitude: np.ndarray,
    cx: float,
    cy: float,
    d: float,
    contrast: float,
) -> tuple[float, float, float]:
    """Return line coherence, closed-edge SNR and normalized edge sharpness."""

    from contour.vision.via_detection.heuristic_detector_parts.score_seed import _fast_percentile
    radius = max(1.0, float(d) * 0.5)
    yy, xx = _coordinate_grids(gradient_magnitude.shape[:2])
    distance = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)

    structure_mask = (distance >= 0.70 * radius) & (distance <= 2.20 * radius)
    sx = gradient_x[structure_mask]
    sy = gradient_y[structure_mask]
    if sx.size:
        jxx = float(np.sum(sx * sx))
        jxy = float(np.sum(sx * sy))
        jyy = float(np.sum(sy * sy))
        trace = jxx + jyy
        coherence = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy)) / (trace + 1e-6)
    else:
        coherence = 0.0

    edge_ring = gradient_magnitude[(distance >= 0.65 * radius) & (distance <= 1.35 * radius)]
    far_ring = gradient_magnitude[(distance >= 1.60 * radius) & (distance <= 2.80 * radius)]
    edge_level = _fast_percentile(edge_ring, 60.0)
    noise_level = _fast_percentile(far_ring, 50.0)
    edge_snr = edge_level / (noise_level + 2.0)
    # A genuine contact has a localized boundary. A broad illumination spot
    # can have high center contrast but its edge changes too slowly at the
    # requested via radius.
    edge_sharpness = (edge_level / 4.0) / max(1.0, abs(float(contrast)))
    return float(coherence), float(edge_snr), float(edge_sharpness)

def _candidate_bbox(
    center_x: float,
    center_y: float,
    diameter: float,
) -> tuple[int, int, int, int]:
    """Return a visible global bounding box for accepted and rejected candidates."""

    size = max(1, round(float(diameter)))
    return (
        round(float(center_x) - size * 0.5),
        round(float(center_y) - size * 0.5),
        size,
        size,
    )
