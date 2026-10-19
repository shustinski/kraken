"""bright_tophat_dog: refine center bright core."""
from __future__ import annotations
from typing import TYPE_CHECKING
from math import pi
import cv2
import numpy as np

from contour.vision.via.bright_tophat_dog_parts.build_isolation_map import _disk_mean_f32

from contour.vision.via.bright_tophat_dog_parts.local_maxima_points import _local_maxima_points

from contour.vision.via.bright_tophat_dog_parts.via_matched_filter_kernel import _percentile_mask

def _stage1_raw_candidates(
    *,
    gray_u8: np.ndarray,
    tophat_u8: np.ndarray,
    dog_u8: np.ndarray,
    matched_u8: np.ndarray,
    tophat_mask: np.ndarray,
    dog_mask: np.ndarray,
    matched_mask: np.ndarray,
    via_mask: np.ndarray,
    cfg: BrightViaDetectorConfig,
    min_area: float,
    max_area: float,
) -> list[_RawStage1]:
    raws: list[_RawStage1] = []
    pi_v = float(pi)
    tophat_f32 = tophat_u8.astype(np.float32, copy=False)
    dog_f32 = dog_u8.astype(np.float32, copy=False)
    # Gather every source first. The score-ranked NMS after this stage resolves
    # duplicates without imposing a count-dependent cutoff or source-order bias.
    for cx, cy, val in _local_maxima_points(matched_u8, matched_mask):
        raw = _raw_from_peak(cx, cy, val, matched_u8, "matched_peak", cfg, min_area, max_area, pi_v)
        if raw is not None:
            raws.append(raw)
    for contour, tag in _contours_from_mask(via_mask, min_area, max_area):
        det = _raw_from_contour(contour, tophat_f32, dog_f32, tag, pi_v)
        if det is not None:
            raws.append(det)
    for label, tmask, response in (
        ("tophat_peak", tophat_mask, tophat_u8),
        ("dog_peak", dog_mask, dog_u8),
        ("gray_peak", _percentile_mask(gray_u8, 95.0), gray_u8),
    ):
        for cx, cy, val in _local_maxima_points(response, tmask):
            raw = _raw_from_peak(cx, cy, val, response, label, cfg, min_area, max_area, pi_v)
            if raw is not None:
                raws.append(raw)
    return raws

def _contours_from_mask(via_mask: np.ndarray, min_area: float, max_area: float) -> list[tuple[np.ndarray, str]]:
    contours, _h = cv2.findContours(via_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[np.ndarray, str]] = []
    for c in contours:
        a = float(cv2.contourArea(c))
        if a > max_area * 1.2 or a < 2.0:
            continue
        out.append((c, "contour"))
    return out

def _raw_from_contour(
    contour: np.ndarray,
    tophat_f32: np.ndarray,
    dog_f32: np.ndarray,
    tag: str,
    pi_v: float,
) -> _RawStage1 | None:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import _RawStage1
    area = float(cv2.contourArea(contour))
    per = float(cv2.arcLength(contour, True))
    if per <= 1e-6 or area <= 1.0:
        return None
    circ = float(4.0 * pi_v * area / (per * per))
    x, y, w, h = cv2.boundingRect(contour)
    aspect = float(w) / float(max(1, h))
    center = _contour_center_from_rect(contour, x, y, w, h)
    top = _disk_mean_f32(tophat_f32, center, max(1.0, (w + h) * 0.2))
    dog = _disk_mean_f32(dog_f32, center, max(1.0, (w + h) * 0.2))
    prelim = max(top, dog) * max(0.2, min(1.0, circ * 1.2))
    return _RawStage1(
        center=center,
        contour=contour,
        bbox=(int(x), int(y), int(w), int(h)),
        area=area,
        circularity=circ,
        aspect=aspect,
        prelim=float(prelim),
        source=tag,
    )

def _raw_from_peak(
    cx: float,
    cy: float,
    val: float,
    _response: np.ndarray,
    label: str,
    cfg: BrightViaDetectorConfig,
    min_area: float,
    max_area: float,
    pi_v: float,
) -> _RawStage1 | None:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import _RawStage1
    d = (float(cfg.diameter_min) + float(cfg.diameter_max)) * 0.5
    r = max(1.0, d * 0.5)
    side = max(2, round(d))
    ix = round(cx) - side // 2
    iy = round(cy) - side // 2
    area = pi_v * r * r
    if not (min_area * 0.5 <= area <= max_area * 1.2):
        return None
    return _RawStage1(
        center=(float(cx), float(cy)),
        contour=None,
        bbox=(ix, iy, side, side),
        area=float(area),
        circularity=1.0,
        aspect=1.0,
        prelim=float(val),
        source=label,
    )

def _contour_center_from_rect(
    contour: np.ndarray, x_coord: int, y_coord: int, width: int, height: int
) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(float(moments.get("m00", 0.0))) > 1e-6:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    return float(x_coord + width * 0.5), float(y_coord + height * 0.5)

def _contour_geometry_center(raw: _RawStage1) -> tuple[float, float]:
    if raw.contour is None:
        return raw.center
    x_coord, y_coord, width, height = raw.bbox
    bbox_center = (x_coord + 0.5 * max(0, width - 1), y_coord + 0.5 * max(0, height - 1))
    rect_center = tuple(float(value) for value in cv2.minAreaRect(raw.contour)[0])
    circle_center_raw, _radius = cv2.minEnclosingCircle(raw.contour)
    circle_center = tuple(float(value) for value in circle_center_raw)
    return (
        float(np.median([bbox_center[0], rect_center[0], circle_center[0]])),
        float(np.median([bbox_center[1], rect_center[1], circle_center[1]])),
    )

def _refine_center_on_bright_core(
    gray_f32: np.ndarray, center: tuple[float, float], diameter: float
) -> tuple[float, float]:
    """Snap the candidate center onto the intensity-weighted centroid of the bright core.

    The stage-1 center is the centroid of a percentile-threshold blob, which drifts
    off the real via when the blob is irregular or merges with neighbouring bright
    structures. Re-centering on the brightest core (top intensities inside a small
    window) removes that drift without depending on the thresholded shape.
    """

    height, width = gray_f32.shape[:2]
    search_r = max(2, round(float(diameter) * 0.6))
    cx0 = round(center[0])
    cy0 = round(center[1])
    x0 = max(0, cx0 - search_r)
    x1 = min(width, cx0 + search_r + 1)
    y0 = max(0, cy0 - search_r)
    y1 = min(height, cy0 + search_r + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return center
    patch = gray_f32[y0:y1, x0:x1]
    weights = patch - float(patch.min())
    peak = float(weights.max())
    if peak <= 1e-6:
        return center
    # Keep only the bright core so a neighbouring spot inside the window cannot pull
    # the centroid away from the via being measured.
    weights = np.where(weights >= peak * 0.5, weights, 0.0)
    total = float(weights.sum())
    if total <= 1e-6:
        return center
    ys, xs = np.mgrid[y0:y1, x0:x1]
    refined_x = float((weights * xs).sum() / total)
    refined_y = float((weights * ys).sum() / total)
    drift_sq = (refined_x - center[0]) ** 2 + (refined_y - center[1]) ** 2
    if drift_sq > float(search_r * search_r):
        return center
    return (refined_x, refined_y)

def _fixed_size_bbox(center: tuple[float, float], diameter: float, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Square bbox of side ``diameter`` centered on ``center`` (clamped to image)."""

    side = max(1, round(float(diameter)))
    half = side / 2.0
    x = round(center[0] - half)
    y = round(center[1] - half)
    if len(shape) >= 2:
        max_x = max(0, int(shape[1]) - side)
        max_y = max(0, int(shape[0]) - side)
        x = min(max(0, x), max_x)
        y = min(max(0, y), max_y)
    return (x, y, side, side)

if TYPE_CHECKING:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import _RawStage1
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detector_config import BrightViaDetectorConfig
