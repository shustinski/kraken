"""heuristic_detector: intensity range seeds."""
from __future__ import annotations
import math
import cv2
import numpy as np
from contour.vision.via_detection.result import ViaDetection

def _local_extrema_seeds(
    response: np.ndarray,
    kernel: np.ndarray,
    pctl: float,
    min_peak: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    dil = cv2.dilate(response, kernel)
    lm = (response == dil) & (response > 0)
    th = max(float(min_peak), float(np.percentile(response, pctl)))
    lm = lm & (response >= th)
    m = lm.astype(np.uint8)
    pts: list[tuple[int, int]] = []
    if int(cv2.countNonZero(m)) > 0:
        nlab, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        for label in range(1, nlab):
            x, y, ww, hh, area = stats[label]
            if int(area) <= 0:
                continue
            roi = response[y : y + hh, x : x + ww]
            lab_roi = labels[y : y + hh, x : x + ww] == label
            if roi.size == 0 or not bool(np.any(lab_roi)):
                cx, cy = centroids[label]
                pts.append((round(cy), round(cx)))
                continue
            masked = np.where(lab_roi, roi, 0)
            peak_value = int(np.max(masked))
            peak_y, peak_x = np.nonzero(lab_roi & (roi == peak_value))
            if peak_x.size:
                pts.append((round(y + float(np.mean(peak_y))), round(x + float(np.mean(peak_x)))))
            else:
                yy, xx = np.unravel_index(int(np.argmax(masked)), masked.shape)
                pts.append((int(y + yy), int(x + xx)))
    return pts, m * 255

def _intensity_range_seeds(
    gray: np.ndarray,
    kernel: np.ndarray,
    value_min: float,
    value_max: float,
    diameter_min: int,
    diameter_max: int,
    *,
    bright: bool,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    lo = min(float(value_min), float(value_max))
    hi = max(float(value_min), float(value_max))
    gray_f32 = gray.astype(np.float32)
    in_range = (gray_f32 >= lo) & (gray_f32 <= hi)
    extreme = gray == (cv2.dilate(gray, kernel) if bright else cv2.erode(gray, kernel))
    support = (in_range & extreme).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(support, connectivity=8)
    min_area = max(2.0, math.pi * (max(1, diameter_min) * 0.5) ** 2 * 0.15)
    max_area = math.pi * (max(1, diameter_max) * 0.5) ** 2 * 2.2
    max_span = max(3, round(float(diameter_max) * 2.0))
    points: list[tuple[int, int]] = []
    accepted_labels = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        _x, _y, width, height, area = stats[label]
        if not (min_area <= float(area) <= max_area):
            continue
        if int(width) > max_span or int(height) > max_span:
            continue
        aspect = max(float(width), float(height)) / max(1.0, min(float(width), float(height)))
        if aspect > 3.2:
            continue
        cx, cy = centroids[label]
        points.append((round(cy), round(cx)))
        accepted_labels[label] = 255
    filtered = accepted_labels[labels]
    return points, filtered

def _grid_suppressed_points(
    pts: list[tuple[int, int]],
    min_dist: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    if not pts:
        return []
    radius = max(1, int(min_dist))
    cell = max(1, radius)
    d2 = float(radius * radius)
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
    out: list[tuple[int, int]] = []
    for py, px in sorted(set(pts), key=lambda p: p[0] * max(1, w) + p[1]):
        cy = max(0, min(h - 1, int(py))) // cell
        cx = max(0, min(w - 1, int(px))) // cell
        too_close = False
        for gy in range(cy - 1, cy + 2):
            for gx in range(cx - 1, cx + 2):
                for qy, qx in grid.get((gy, gx), ()):
                    if (py - qy) ** 2 + (px - qx) ** 2 < d2:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            continue
        point = (int(py), int(px))
        out.append(point)
        grid.setdefault((cy, cx), []).append(point)
    return out

def _merge_seeds(allp: list[tuple[int, int]], min_dist: int, h: int, w: int) -> list[tuple[int, int]]:
    return _grid_suppressed_points(allp, min_dist, h, w)

def _spread_points(pts: list[tuple[int, int]], min_dist: int, h: int, w: int) -> list[tuple[int, int]]:
    return _grid_suppressed_points(pts, min_dist, h, w)

def _dedupe_by_score(dets: list[ViaDetection], min_dist: float) -> list[ViaDetection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    radius = max(0.5, float(min_dist))
    cell = max(1.0, radius)
    d2 = float(radius * radius)
    kept: list[ViaDetection] = []
    grid: dict[tuple[int, int], list[ViaDetection]] = {}
    for d in dets:
        if d.reject_reason and "hard" in d.reject_reason:
            kept.append(d)
            continue
        cx = int(d.x // cell)
        cy = int(d.y // cell)
        too_close = False
        for gy in range(cy - 1, cy + 2):
            for gx in range(cx - 1, cx + 2):
                if any((d.x - k.x) ** 2 + (d.y - k.y) ** 2 < d2 for k in grid.get((gy, gx), ())):
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            continue
        kept.append(d)
        grid.setdefault((cy, cx), []).append(d)
    return kept

def _nms_simple(dets: list[ViaDetection], dist: int) -> list[ViaDetection]:
    hard = [d for d in dets if d.reject_reason and "hard" in str(d.reject_reason)]
    soft = [d for d in dets if not (d.reject_reason and "hard" in str(d.reject_reason))]
    soft.sort(key=lambda x: x.score, reverse=True)
    d2 = float(max(0, int(dist)) ** 2) if dist > 0 else 0.0
    keep: list[ViaDetection] = []
    grid: dict[tuple[int, int], list[ViaDetection]] = {}
    cell = max(1, int(dist))
    for d in soft:
        if d2 == 0.0:
            keep.append(d)
            continue
        cx = int(d.x // cell)
        cy = int(d.y // cell)
        too_close = False
        for gy in range(cy - 1, cy + 2):
            for gx in range(cx - 1, cx + 2):
                if any((d.x - o.x) ** 2 + (d.y - o.y) ** 2 <= d2 for o in grid.get((gy, gx), ())):
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            continue
        keep.append(d)
        grid.setdefault((cy, cx), []).append(d)
    return hard + keep
