"""bright_tophat_dog: local maxima points."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np

from contour.vision.via.bright_tophat_dog_parts.build_isolation_map import _raw_nms_rank

from contour.vision.via.bright_tophat_dog_parts.center_distance_index import _CenterDistanceIndex

def _local_maxima_points(
    response_u8: np.ndarray,
    support: np.ndarray,
) -> list[tuple[float, float, float]]:
    if response_u8.size == 0:
        return []
    work = response_u8.astype(np.float32)
    work[support == 0] = 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dil = cv2.dilate(work, kernel)
    lm = (work >= dil) & (work > 0) & (support > 0)
    maxima = lm.astype(np.uint8)
    if int(cv2.countNonZero(maxima)) == 0:
        return []
    # A saturated/flat peak is one maximum, not one candidate per pixel. Collapse
    # each connected plateau while retaining every spatially distinct maximum.
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(maxima, connectivity=8)
    points: list[tuple[float, float, float]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if int(area) <= 0:
            continue
        roi = work[y : y + height, x : x + width]
        region = labels[y : y + height, x : x + width] == label
        peak = float(np.max(roi[region]))
        peak_y, peak_x = np.nonzero(region & (roi == peak))
        points.append(
            (
                float(x) + float(np.mean(peak_x)),
                float(y) + float(np.mean(peak_y)),
                peak,
            )
        )
    points.sort(key=lambda point: point[2], reverse=True)
    return points

def _raw_nms(raws: list[_RawStage1], min_dist: float) -> list[_RawStage1]:
    if not raws or min_dist <= 0:
        return raws
    kept: list[_RawStage1] = []
    idx = _CenterDistanceIndex(float(min_dist))
    for r in sorted(raws, key=lambda z: _raw_nms_rank(z), reverse=True):
        cx, cy = r.center
        if idx.is_close(cx, cy):
            continue
        kept.append(r)
        idx.add(cx, cy)
    return kept

def _passes_brightness_gate(center_mean: float, cfg: BrightViaDetectorConfig) -> bool:
    if cfg.bright_range_enabled or cfg.dark_range_enabled:
        bright_ok = cfg.bright_range_enabled and float(cfg.bright_range_min) <= center_mean <= float(
            cfg.bright_range_max
        )
        dark_ok = cfg.dark_range_enabled and float(cfg.dark_range_min) <= center_mean <= float(cfg.dark_range_max)
        return bright_ok or dark_ok
    return center_mean >= float(cfg.bright_center_min_score) * 0.999

def _brightness_gate_reason(center_mean: float, cfg: BrightViaDetectorConfig) -> str:
    if cfg.bright_range_enabled or cfg.dark_range_enabled:
        parts: list[str] = []
        if cfg.bright_range_enabled:
            parts.append(f"bright={cfg.bright_range_min:.0f}-{cfg.bright_range_max:.0f}")
        if cfg.dark_range_enabled:
            parts.append(f"dark={cfg.dark_range_min:.0f}-{cfg.dark_range_max:.0f}")
        return f"brightness_out_of_range (raw={center_mean:.1f}, {'; '.join(parts)})"
    return f"low_center (raw={center_mean:.1f} min={cfg.bright_center_min_score})"

if TYPE_CHECKING:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import _RawStage1
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detector_config import BrightViaDetectorConfig
