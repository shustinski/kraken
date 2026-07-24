"""Heuristic via detection: local extrema + local component + structural scoring (no top-hat/DoG/ML)."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import cv2
import numpy as np

from .binary_mask import detect_binary_components, looks_like_binary_mask
from .config import HeuristicViaDetectorConfig, ViaPolarity
from .result import DetectionResult, ViaDetection


class ViaRejectCode(StrEnum):
    BRIGHTNESS_RANGE = "brightness_range"
    LOW_CENTER_BRIGHTNESS = "low_center_brightness"
    LOW_CONTRAST = "low_contrast"
    LOW_PROMINENCE = "low_prominence"
    NO_COMPONENT = "no_component"
    SIZE_MISMATCH = "size_mismatch"
    CENTER_DRIFT = "center_drift"
    LOW_COMPACTNESS = "compact"
    LOW_CIRCULARITY = "circularity"
    ELONGATION = "elongation"
    LINE_COHERENCE = "line_coherence"
    DIFFUSE_SPOT = "diffuse_spot"


@dataclass(frozen=True)
class PreparedHeuristicFrame:
    gray: np.ndarray
    gray_f32: np.ndarray
    corrected_u8: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    gradient_magnitude: np.ndarray


@dataclass(frozen=True)
class CandidateFeatures:
    center_x: float
    center_y: float
    diameter: float
    contrast: float
    prominence: float
    compactness: float
    circularity: float
    aspect: float
    line_coherence: float
    edge_snr: float
    edge_sharpness: float
    border_imbalance: float
    center_brightness: float = 0.0
    equivalent_diameter: float = 0.0
    center_drift: float = 0.0
    binarization_threshold: float = 0.0


@dataclass(frozen=True)
class SegmentedCandidate:
    gray: np.ndarray
    median: float
    contrast: float
    prominence: float
    area: float
    component_mask: np.ndarray
    contour: np.ndarray
    component_stats: np.ndarray
    binarization_threshold: float


def _hard_reason(code: ViaRejectCode, detail: str = "") -> str:
    return f"hard:{code.value}{detail}"


def _det_better(candidate: ViaDetection, current: ViaDetection | None) -> bool:
    if current is None:
        return True
    c_hard = bool(current.reject_reason and str(current.reject_reason).startswith("hard:"))
    n_hard = bool(candidate.reject_reason and str(candidate.reject_reason).startswith("hard:"))
    if c_hard and not n_hard:
        return True
    if n_hard and not c_hard:
        return False
    return float(candidate.score) > float(current.score)


def detect_vias_heuristic(image: np.ndarray, config: HeuristicViaDetectorConfig) -> DetectionResult:
    started_at = time.perf_counter()
    config.validate()
    g = _to_gray_u8(image)
    h, w = g.shape[:2]
    snap0 = dict(config.snapshot())
    if h < 3 or w < 3:
        return DetectionResult(method="heuristic", accepted=[], parameters_snapshot=snap0)

    allowed = config.allowed_diameters()
    if not allowed:
        return DetectionResult(method="heuristic", accepted=[], parameters_snapshot=snap0)
    d_min, d_max = min(allowed), max(allowed)
    if looks_like_binary_mask(g):
        _mask, components = detect_binary_components(
            g,
            diameter_min=d_min,
            diameter_max=d_max,
            min_area_factor=0.35,
            max_area_factor=2.0,
            min_aspect=1.0 / max(1.0, float(config.max_elongation)),
            max_aspect=max(1.0, float(config.max_elongation)),
            nms_distance=config.nms_distance,
        )
        accepted = [
            ViaDetection(
                x=item.center[0],
                y=item.center[1],
                bbox=item.bbox,
                score=100.0,
                diameter_estimate=0.5 * (item.bbox[2] + item.bbox[3]),
                contrast=255.0,
                prominence=255.0,
                compactness=item.circularity,
                aspect=item.aspect,
                polarity_hypothesis="binary",
            )
            for item in components
        ]
        return DetectionResult(
            method="heuristic_binary",
            accepted=accepted,
            debug_images={"binary_mask": _mask},
            parameters_snapshot={**snap0, "binary_mask": True, "accepted_count": len(accepted)},
        )

    prepare_started = time.perf_counter()
    seed_threshold = {"percentile": float(config.seed_percentile)}
    g_pre = _preprocess_denoise(g, config)
    bg = cv2.GaussianBlur(g_pre, (0, 0), float(config.background_sigma))
    gray_f32 = g_pre.astype(np.float32)
    corr = gray_f32 - bg.astype(np.float32)
    corr_u8 = _normalize01_to_u8(corr)
    gradient_x = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    frame = PreparedHeuristicFrame(
        gray=g_pre,
        gray_f32=gray_f32,
        corrected_u8=corr_u8,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        gradient_magnitude=cv2.magnitude(gradient_x, gradient_y),
    )
    prepare_seconds = time.perf_counter() - prepare_started

    bright_map = corr_u8
    dark_map = 255 - corr_u8

    ksize = int(max(3, 2 * int(round(0.5 * d_max)) + 1))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    seeds_b, mask_b = _local_extrema_seeds(
        bright_map,
        ker,
        seed_threshold["percentile"],
        float(config.min_peak_grey),
    )
    seeds_d, mask_d = _local_extrema_seeds(
        dark_map,
        ker,
        seed_threshold["percentile"],
        float(config.min_peak_grey),
    )

    # An explicitly enabled intensity range is a candidate-generation rule, not
    # merely a late validation gate. Otherwise dense frames keep only the top
    # response percentile and valid, slightly weaker vias never get scored.
    if config.bright_range_enabled and (
        float(config.bright_range_min) >= 128.0 or config.use_intensity_range_seeds
    ):
        range_seeds, range_mask = _intensity_range_seeds(
            g_pre,
            ker,
            float(config.bright_range_min),
            float(config.bright_range_max),
            d_min,
            d_max,
            bright=True,
        )
        seeds_b.extend(range_seeds)
        mask_b = cv2.bitwise_or(mask_b, range_mask)
    if config.dark_range_enabled and (
        float(config.dark_range_max) <= 127.0 or config.use_intensity_range_seeds
    ):
        range_seeds, range_mask = _intensity_range_seeds(
            g_pre,
            ker,
            float(config.dark_range_min),
            float(config.dark_range_max),
            d_min,
            d_max,
            bright=False,
        )
        seeds_d.extend(range_seeds)
        mask_d = cv2.bitwise_or(mask_d, range_mask)

    polar = str(config.polarity or ViaPolarity.AUTO).lower()
    min_sep = int(config.min_distance_between_peaks) if config.min_distance_between_peaks else max(2, d_min // 2)
    if polar in (str(ViaPolarity.BRIGHT), "bright"):
        raw_seeds = _spread_points(seeds_b, min_sep, h, w)
    elif polar in (str(ViaPolarity.DARK), "dark"):
        raw_seeds = _spread_points(seeds_d, min_sep, h, w)
    else:
        raw_seeds = _merge_seeds(list(seeds_b) + list(seeds_d), min_sep, h, w)
    hyps: list[str]
    if polar in ("auto", str(ViaPolarity.AUTO)):
        hyps = []
        if config.bright_range_enabled:
            hyps.append(str(ViaPolarity.BRIGHT))
        if config.dark_range_enabled:
            hyps.append(str(ViaPolarity.DARK))
        if not hyps:
            hyps = [str(ViaPolarity.BRIGHT)]
    elif polar in (
        str(ViaPolarity.RING_LIGHT_RING),
        str(ViaPolarity.RING_DARK_RING),
        ViaPolarity.RING_LIGHT_RING,
        ViaPolarity.RING_DARK_RING,
    ):
        hyps = [polar] if not isinstance(polar, ViaPolarity) else [str(polar)]
    else:
        hyps = [polar]

    score_started = time.perf_counter()
    dets = [
        item
        for seed in raw_seeds
        if (item := _score_seed(frame, seed, allowed, hyps, config)) is not None
    ]

    scored_count = len(dets)
    dets = [d for d in dets if d.reject_reason is None or (d.reject_reason and "hard" in d.reject_reason)]
    dets = _dedupe_by_score(dets, min_dist=1.0)
    dets.sort(key=lambda d: d.score, reverse=True)
    effective_nms = max(0, int(config.nms_distance), int(round(0.6 * d_max)))
    after = _nms_simple(dets, effective_nms)

    accepted: list[ViaDetection] = [
        d for d in after if d.reject_reason is None and d.score >= float(config.min_final_score)
    ]
    below: list[ViaDetection] = [
        d for d in after if d.reject_reason is None and 0 < d.score < float(config.min_final_score)
    ]
    hard = [d for d in after if d.reject_reason and "hard" in d.reject_reason]

    scoring_seconds = time.perf_counter() - score_started
    dbg = _debug_viz(
        g_pre,
        corr_u8,
        mask_b,
        mask_d,
        accepted,
        below,
        hard,
        d_max,
    )

    reject_counts = Counter(
        str(item.reject_reason).split("(", 1)[0].removeprefix("hard:") for item in hard if item.reject_reason
    )
    total_seconds = time.perf_counter() - started_at
    return DetectionResult(
        method="heuristic",
        accepted=accepted,
        rejected=hard,
        below_threshold=below,
        debug_images=dbg,
        parameters_snapshot={
            **snap0,
            "raw_seed_count": len(raw_seeds),
            "scored_candidate_count": scored_count,
            "candidate_count_after_dedupe": len(dets),
            "candidate_count_after_nms": len(after),
            "effective_nms_distance": effective_nms,
            "accepted_count": len(accepted),
            "below_threshold_count": len(below),
            "hard_rejected_count": len(hard),
            "reject_counts": dict(sorted(reject_counts.items())),
            "seed_threshold": seed_threshold,
            "timings_seconds": {
                "prepare": prepare_seconds,
                "score": scoring_seconds,
                "total": total_seconds,
            },
            "scoring_workers": 1,
        },
    )


def analyze_via_at(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    config: HeuristicViaDetectorConfig,
) -> ViaDetection | None:
    """Measure one caller-supplied contact center with the heuristic scorer."""

    return analyze_vias_at(image, [(center_x, center_y)], config)[0]


def analyze_vias_at(
    image: np.ndarray,
    centers: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    config: HeuristicViaDetectorConfig,
) -> list[ViaDetection | None]:
    """Measure caller-supplied contact centers while preparing the frame once."""

    config.validate()
    gray = _to_gray_u8(np.asarray(image))
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return [None] * len(centers)
    allowed = config.allowed_diameters()
    if not allowed:
        return [None] * len(centers)
    frame = _prepare_heuristic_frame(gray, config)
    hypotheses = _candidate_hypotheses(config)
    results: list[ViaDetection | None] = []
    for center_x, center_y in centers:
        x_coord = max(0, min(width - 1, int(round(float(center_x)))))
        y_coord = max(0, min(height - 1, int(round(float(center_y)))))
        results.append(
            _score_seed(
                frame,
                (y_coord, x_coord),
                allowed,
                hypotheses,
                config,
                enforce_rejections=False,
            )
        )
    return results


def _prepare_heuristic_frame(
    gray: np.ndarray,
    config: HeuristicViaDetectorConfig,
) -> PreparedHeuristicFrame:
    denoised = _preprocess_denoise(_to_gray_u8(gray), config)
    background = cv2.GaussianBlur(denoised, (0, 0), float(config.background_sigma))
    gray_f32 = denoised.astype(np.float32)
    corrected = gray_f32 - background.astype(np.float32)
    corrected_u8 = _normalize01_to_u8(corrected)
    gradient_x = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    return PreparedHeuristicFrame(
        gray=denoised,
        gray_f32=gray_f32,
        corrected_u8=corrected_u8,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        gradient_magnitude=cv2.magnitude(gradient_x, gradient_y),
    )


def _candidate_hypotheses(config: HeuristicViaDetectorConfig) -> list[str]:
    polarity = str(config.polarity or ViaPolarity.AUTO).lower()
    if polarity in ("auto", str(ViaPolarity.AUTO)):
        hypotheses: list[str] = []
        if config.bright_range_enabled:
            hypotheses.append(str(ViaPolarity.BRIGHT))
        if config.dark_range_enabled:
            hypotheses.append(str(ViaPolarity.DARK))
        return hypotheses or [str(ViaPolarity.BRIGHT)]
    if polarity in {
        str(ViaPolarity.RING_LIGHT_RING),
        str(ViaPolarity.RING_DARK_RING),
    }:
        return [polarity]
    return [polarity]


def _score_seed(
    frame: PreparedHeuristicFrame,
    seed: tuple[int, int],
    allowed_diameters: list[int],
    hypotheses: list[str],
    config: HeuristicViaDetectorConfig,
    *,
    enforce_rejections: bool = True,
) -> ViaDetection | None:
    """Score every allowed diameter for one seed and return its best candidate."""

    cy, cx = seed
    height, width = frame.gray.shape[:2]
    patch_scale = max(1.0, float(config.analysis_window_scale))
    min_patch_size = int(config.min_analyze_size)
    best: ViaDetection | None = None
    for diameter in allowed_diameters:
        patch_size = int(max(min_patch_size, round(patch_scale * float(diameter))))
        half = patch_size // 2
        y0, y1 = max(0, cy - half), min(height, cy + half + 1)
        x0, x1 = max(0, cx - half), min(width, cx + half + 1)
        patch = frame.gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        patch_x, patch_y = cx - x0, cy - y0
        for hypothesis in hypotheses:
            if hypothesis in {str(ViaPolarity.AUTO), "auto"}:
                continue
            detection = _score_one(
                patch,
                patch_x,
                patch_y,
                diameter,
                (x0, y0),
                hypothesis,
                config,
                frame.gradient_x[y0:y1, x0:x1],
                frame.gradient_y[y0:y1, x0:x1],
                frame.gradient_magnitude[y0:y1, x0:x1],
                allowed_diameters,
                enforce_rejections=enforce_rejections,
            )
            if detection is not None and _det_better(detection, best):
                best = detection
    return best


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[:, :, 0].astype(np.uint8, copy=False)


def _preprocess_denoise(g: np.ndarray, config: HeuristicViaDetectorConfig) -> np.ndarray:
    if bool(config.use_bilateral):
        return cv2.bilateralFilter(
            g, int(config.bilateral_d), float(config.bilateral_sigma_color), float(config.bilateral_sigma_space)
        )
    return cv2.medianBlur(g, 3)


def _normalize01_to_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo + 1e-6:
        return (np.zeros_like(a) + 128.0).astype(np.uint8)
    u = ((a - lo) / (hi - lo) * 255.0).clip(0, 255)
    return u.astype(np.uint8)


def _fast_percentile_u8(values: np.ndarray, percentile: float) -> float:
    """Linear percentile for uint8 patches without NumPy's generic quantile setup."""

    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    rank = (flat.size - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    selected = np.partition(flat, (lower, upper))
    if lower == upper:
        return float(selected[lower])
    fraction = rank - lower
    return float(selected[lower]) * (1.0 - fraction) + float(selected[upper]) * fraction


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
                pts.append((int(round(cy)), int(round(cx))))
                continue
            masked = np.where(lab_roi, roi, 0)
            peak_value = int(np.max(masked))
            peak_y, peak_x = np.nonzero(lab_roi & (roi == peak_value))
            if peak_x.size:
                pts.append((int(round(y + float(np.mean(peak_y)))), int(round(x + float(np.mean(peak_x))))))
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
    max_span = max(3, int(round(float(diameter_max) * 2.0)))
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
        points.append((int(round(cy)), int(round(cx))))
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
def _coordinate_grids(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
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
    v = patch[m]
    if v.size == 0:
        return float(np.mean(patch))
    return float(np.mean(v))


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
    edge_level = float(np.percentile(edge_ring, 60.0)) if edge_ring.size else 0.0
    noise_level = float(np.median(far_ring)) if far_ring.size else 0.0
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


def _segment_candidate(
    patch: np.ndarray,
    pcx: int,
    pcy: int,
    diameter: int,
    offset: tuple[int, int],
    hypothesis: str,
    config: HeuristicViaDetectorConfig,
    *,
    enforce_rejections: bool = True,
) -> tuple[SegmentedCandidate | None, ViaDetection | None]:
    """Build the local component and return either its data or a typed rejection."""

    height, width = patch.shape[:2]
    center_mask, inner_mask, outer_mask = _annulus_masks(
        (height, width), pcx, pcy, float(diameter)
    )
    center_grey = float(patch[pcy, pcx])

    def rejected(code: ViaRejectCode, *, contrast: float = 0.0, prominence: float = 0.0) -> ViaDetection:
        return ViaDetection(
            float(offset[0] + pcx),
            float(offset[1] + pcy),
            _candidate_bbox(offset[0] + pcx, offset[1] + pcy, diameter),
            0.0,
            float(diameter),
            float(contrast),
            float(prominence),
            0.0,
            0.0,
            hypothesis,
            _hard_reason(code),
        )

    if enforce_rejections and not _center_in_brightness_range(center_grey, hypothesis, config):
        return None, rejected(ViaRejectCode.BRIGHTNESS_RANGE)
    if enforce_rejections and center_grey < float(config.min_center_brightness):
        return None, rejected(ViaRejectCode.LOW_CENTER_BRIGHTNESS)
    annular_contrast = _contrast_for_polarity(
        patch, center_mask, inner_mask, outer_mask, hypothesis
    )
    background_contrast = _local_center_background_contrast(
        patch, pcx, pcy, float(diameter), hypothesis
    )
    contrast = max(float(annular_contrast), float(background_contrast))
    if enforce_rejections and contrast < float(config.min_center_contrast):
        return None, rejected(ViaRejectCode.LOW_CONTRAST, contrast=contrast)

    median = _fast_percentile_u8(patch, 50.0)
    peak_radius = min(4, max(1, max(height, width) // 6))
    y0, y1 = max(0, pcy - peak_radius), min(height, pcy + peak_radius + 1)
    x0, x1 = max(0, pcx - peak_radius), min(width, pcx + peak_radius + 1)
    neighbourhood = patch[y0:y1, x0:x1].ravel()
    prominence = float(np.max(np.abs(neighbourhood - median))) if neighbourhood.size else 0.0
    if enforce_rejections and prominence < float(config.min_peak_prominence):
        return None, rejected(
            ViaRejectCode.LOW_PROMINENCE,
            contrast=contrast,
            prominence=prominence,
        )

    percentile = float(config.local_binarize_percentile)
    segmentation_contrast_floor = (
        float(config.min_center_contrast)
        if enforce_rejections or contrast >= float(config.min_center_contrast)
        else 0.0
    )
    delta = max(segmentation_contrast_floor, abs(contrast) * 0.30, 2.0)
    dark_component = hypothesis in (
        str(ViaPolarity.DARK),
        "dark",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    )
    ring_component = hypothesis in (
        str(ViaPolarity.RING_LIGHT_RING),
        "ring_light_ring",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    )
    clamp_dark_seed = enforce_rejections and (
        config.use_intensity_range_seeds or float(config.dark_range_max) <= 127.0
    )
    clamp_bright_seed = enforce_rejections and (
        config.use_intensity_range_seeds or float(config.bright_range_min) >= 128.0
    )
    if dark_component:
        threshold = min(
            _fast_percentile_u8(patch, max(1.0, 100.0 - percentile)),
            median - delta,
        )
        if config.dark_range_enabled and clamp_dark_seed:
            threshold = min(threshold, float(config.dark_range_max))
            if not ring_component and clamp_dark_seed:
                threshold = max(center_grey, threshold)
                if center_grey < threshold:
                    threshold = max(center_grey, math.ceil(threshold) - 1.0)
            binary = (
                (patch >= float(config.dark_range_min)) & (patch <= threshold)
            ).astype(np.uint8) * 255
        else:
            if not ring_component and clamp_dark_seed:
                threshold = max(center_grey, threshold)
                if center_grey < threshold:
                    threshold = max(center_grey, math.ceil(threshold) - 1.0)
            binary = (patch <= threshold).astype(np.uint8) * 255
    else:
        threshold = max(_fast_percentile_u8(patch, percentile), median + delta)
        if (
            hypothesis in (str(ViaPolarity.BRIGHT), "bright")
            and config.bright_range_enabled
            and clamp_bright_seed
        ):
            threshold = max(threshold, float(config.bright_range_min))
            if not ring_component and clamp_bright_seed:
                threshold = min(center_grey, threshold)
                if center_grey > threshold:
                    threshold = min(center_grey, math.floor(threshold) + 1.0)
            binary = (
                (patch >= threshold) & (patch <= float(config.bright_range_max))
            ).astype(np.uint8) * 255
        else:
            if not ring_component and clamp_bright_seed:
                threshold = min(center_grey, threshold)
                if center_grey > threshold:
                    threshold = min(center_grey, math.floor(threshold) + 1.0)
            binary = (patch >= threshold).astype(np.uint8) * 255

    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if hypothesis in (
        str(ViaPolarity.RING_LIGHT_RING),
        "ring_light_ring",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    ):
        ring_labels = labels[inner_mask]
        ring_labels = ring_labels[ring_labels > 0]
        if ring_labels.size:
            label = int(np.bincount(ring_labels).argmax())
        else:
            label = 0
    else:
        label = int(labels[pcy, pcx])
    if label <= 0:
        return None, rejected(
            ViaRejectCode.NO_COMPONENT,
            contrast=contrast,
            prominence=prominence,
        )
    area = float(stats[label, cv2.CC_STAT_AREA])
    if area < 1.0:
        return None, None
    component_bool = labels == label
    component_mask = component_bool.astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    return (
        SegmentedCandidate(
            gray=patch,
            median=median,
            contrast=contrast,
            prominence=prominence,
            area=area,
            component_mask=component_bool,
            contour=max(contours, key=cv2.contourArea),
            component_stats=stats[label],
            binarization_threshold=float(threshold),
        ),
        None,
    )


def _score_one(
    patch: np.ndarray,
    pcx: int,
    pcy: int,
    d_est: int,
    offset: tuple[int, int],
    hyp: str,
    config: HeuristicViaDetectorConfig,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    gradient_magnitude: np.ndarray,
    allowed_diameters: list[int],
    *,
    enforce_rejections: bool = True,
) -> ViaDetection | None:
    h, w = patch.shape[:2]
    if pcx < 0 or pcy < 0 or pcx >= w or pcy >= h:
        return None
    ph, pw = h, w
    segmented, rejection = _segment_candidate(
        patch,
        pcx,
        pcy,
        d_est,
        offset,
        hyp,
        config,
        enforce_rejections=enforce_rejections,
    )
    if rejection is not None:
        return rejection
    if segmented is None:
        return None
    gpatch = segmented.gray
    med = segmented.median
    contrast = segmented.contrast
    prom = segmented.prominence
    area = segmented.area
    cnt0 = segmented.contour
    if hyp in (
        str(ViaPolarity.RING_LIGHT_RING),
        "ring_light_ring",
        str(ViaPolarity.RING_DARK_RING),
        "ring_dark_ring",
    ):
        _ring_center, ring_radius = cv2.minEnclosingCircle(cnt0)
        eq_diam = 2.0 * float(ring_radius)
    else:
        eq_diam = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
    tol = config.effective_size_tolerance()
    re = max(float(d_est), 1.0)
    if enforce_rejections and abs(eq_diam - float(d_est)) / re > tol:
        return ViaDetection(
            float(offset[0] + pcx),
            float(offset[1] + pcy),
            _candidate_bbox(offset[0] + pcx, offset[1] + pcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            0.0,
            0.0,
            hyp,
            _hard_reason(ViaRejectCode.SIZE_MISMATCH, f"(eq={eq_diam:.1f},d={d_est})"),
        )
    weighted_center = _refine_center_xy(
        gpatch,
        segmented.component_mask,
        seed_x=float(pcx),
        seed_y=float(pcy),
        med=med,
        hyp=hyp,
    )
    fcx, fcy = _component_geometry_center(cnt0, segmented.component_stats, weighted_center)
    drift = math.hypot(fcx - float(pcx), fcy - float(pcy))
    max_drift = float(config.max_center_drift_ratio) * re
    if enforce_rejections and drift > max_drift:
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            0.0,
            0.0,
            hyp,
            _hard_reason(ViaRejectCode.CENTER_DRIFT, f"({drift:.1f}>{max_drift:.1f})"),
        )
    r_rect = cv2.minAreaRect(cnt0)
    w_r, h_r = float(r_rect[1][0]), float(r_rect[1][1])
    if w_r < 1e-3 or h_r < 1e-3:
        aspect = 1.0
    else:
        aspect = max(w_r, h_r) / (min(w_r, h_r) + 1e-6)
    r_expect = d_est * 0.5
    perim = float(cv2.arcLength(cnt0, True)) + 1e-3
    circ2 = (
        min(1.0, max(0.0, 4.0 * math.pi * max(area, 1.0) / (perim * perim)))
        if perim > 0
        else 0.0
    )
    fill = 4.0 * area / (w_r * h_r + 1e-6) if w_r * h_r > 1e-6 else 0.0
    compact2 = 0.5 * min(1.0, min(fill, 1.0)) + 0.5 * min(1.0, area / (math.pi * r_expect**2 + 1e-3))
    if enforce_rejections and compact2 < float(config.min_compactness):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.LOW_COMPACTNESS),
        )
    if enforce_rejections and circ2 < float(config.min_circularity):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.LOW_CIRCULARITY),
        )
    if enforce_rejections and aspect > float(config.max_elongation):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.ELONGATION),
        )

    structure_coherence, edge_snr, edge_sharpness = _local_structure_metrics(
        gradient_x,
        gradient_y,
        gradient_magnitude,
        fcx,
        fcy,
        float(d_est),
        float(contrast),
    )
    if enforce_rejections and structure_coherence > float(config.max_line_coherence):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.LINE_COHERENCE, f"({structure_coherence:.2f})"),
        )
    if enforce_rejections and edge_sharpness < float(config.min_edge_sharpness):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            _candidate_bbox(offset[0] + fcx, offset[1] + fcy, d_est),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            _hard_reason(ViaRejectCode.DIFFUSE_SPOT, f"({edge_sharpness:.2f})"),
        )

    icx, icy = int(round(fcx)), int(round(fcy))
    icx = max(0, min(pw - 1, icx))
    icy = max(0, min(ph - 1, icy))
    r_edge = int(max(1, d_est // 3))
    ys = int(max(0, icy - r_edge))
    ye = int(min(ph - 1, icy + r_edge))
    xs = int(max(0, icx - r_edge))
    xe = int(min(pw - 1, icx + r_edge))
    left = float(np.mean(gpatch[icy, xs:icx])) if icx > xs else gpatch[icy, icx]
    right = float(np.mean(gpatch[icy, icx : xe + 1])) if icx < xe else gpatch[icy, icx]
    up = float(np.mean(gpatch[ys:icy, icx])) if icy > ys else gpatch[icy, icx]
    down = float(np.mean(gpatch[icy:ye, icx])) if icy < ye else gpatch[icy, icx]
    border_n = (abs(left - right) + abs(up - down)) / 255.0
    el = max(0.0, aspect - 1.0)
    aspect_line_n = min(1.0, el / (float(config.max_elongation) + 0.1))
    coherence_line_n = _scale01(structure_coherence, 0.30, 0.80)
    line_n = max(aspect_line_n, coherence_line_n)

    features = CandidateFeatures(
        center_x=fcx,
        center_y=fcy,
        diameter=float(d_est),
        contrast=float(contrast),
        prominence=float(prom),
        compactness=float(compact2),
        circularity=float(circ2),
        aspect=float(aspect),
        line_coherence=float(structure_coherence),
        edge_snr=float(edge_snr),
        edge_sharpness=float(edge_sharpness),
        border_imbalance=float(border_n),
        center_brightness=float(gpatch[icy, icx]),
        equivalent_diameter=float(eq_diam),
        center_drift=float(drift),
        binarization_threshold=float(segmented.binarization_threshold),
    )
    score_metrics = _candidate_score_metrics(features, config, allowed_diameters, line_n)
    final = score_metrics["final_score"]

    gx = float(offset[0]) + fcx
    gy = float(offset[1]) + fcy
    half = float(d_est) * 0.5
    ox = int(round(gx - half))
    oy = int(round(gy - half))
    bbox = (ox, oy, int(d_est), int(d_est))
    return ViaDetection(
        x=gx,
        y=gy,
        bbox=bbox,
        score=float(final),
        diameter_estimate=float(d_est),
        contrast=float(contrast),
        prominence=float(prom),
        compactness=float(compact2),
        aspect=float(aspect),
        polarity_hypothesis=hyp,
        reject_reason=None,
        features={
            "center_brightness": features.center_brightness,
            "contrast": features.contrast,
            "prominence": features.prominence,
            "diameter": features.diameter,
            "equivalent_diameter": features.equivalent_diameter,
            "center_drift": features.center_drift,
            "compactness": features.compactness,
            "circularity": features.circularity,
            "aspect": features.aspect,
            "line_coherence": features.line_coherence,
            "edge_snr": features.edge_snr,
            "edge_sharpness": features.edge_sharpness,
            "border_imbalance": features.border_imbalance,
            "line_likeness": float(line_n),
            "binarization_threshold": features.binarization_threshold,
            **score_metrics,
        },
    )


def _candidate_score_metrics(
    features: CandidateFeatures,
    config: HeuristicViaDetectorConfig,
    allowed_diameters: list[int],
    line_likeness: float,
) -> dict[str, float]:
    """Return the normalized features and weighted contributions used by the score."""

    sc_contrast = _scale01(features.contrast, config.contrast_score_min, config.contrast_score_max)
    edge_span = 1.0 - float(config.edge_quality_floor)
    edge_quality = float(config.edge_quality_floor) + edge_span * _scale01(
        features.edge_snr,
        config.edge_snr_score_min,
        config.edge_snr_score_max,
    )
    sc_prominence = _scale01(
        features.prominence,
        config.prominence_score_min,
        config.prominence_score_max,
    ) * edge_quality
    diameter_min = float(min(allowed_diameters))
    diameter_max = float(max(allowed_diameters))
    sc_size = 1.0 - min(
        1.0,
        abs(features.diameter - 0.5 * (diameter_min + diameter_max))
        / (diameter_max - diameter_min + 1.0)
        * 0.4,
    )
    sc_compact = min(1.0, max(0.0, features.compactness))
    sc_round = min(1.0, max(0.0, min(features.circularity, 1.0)))
    sc_balance = 1.0 - min(1.0, features.border_imbalance * float(config.border_balance_scale))
    contribution_contrast = float(config.w_contrast) * sc_contrast
    contribution_prominence = float(config.w_prominence) * sc_prominence
    contribution_size = float(config.w_size) * sc_size
    contribution_compact = float(config.w_compact) * sc_compact
    contribution_round = float(config.w_round) * sc_round
    contribution_balance = float(config.w_balance) * sc_balance
    penalty_line = float(config.w_line) * line_likeness * float(config.line_penalty_scale)
    penalty_border = (
        float(config.w_border) * features.border_imbalance * float(config.border_penalty_scale)
    )
    raw = (
        contribution_contrast
        + contribution_prominence
        + contribution_size
        + contribution_compact
        + contribution_round
        + contribution_balance
        - penalty_line
        - penalty_border
    )
    return {
        "normalized_contrast": float(sc_contrast),
        "normalized_prominence": float(sc_prominence),
        "normalized_size": float(sc_size),
        "normalized_compactness": float(sc_compact),
        "normalized_roundness": float(sc_round),
        "normalized_balance": float(sc_balance),
        "edge_quality": float(edge_quality),
        "contribution_contrast": float(contribution_contrast),
        "contribution_prominence": float(contribution_prominence),
        "contribution_size": float(contribution_size),
        "contribution_compactness": float(contribution_compact),
        "contribution_roundness": float(contribution_round),
        "contribution_balance": float(contribution_balance),
        "penalty_line": float(penalty_line),
        "penalty_border": float(penalty_border),
        "raw_score": float(raw),
        "final_score": max(0.0, min(100.0, float(raw))),
    }


def _final_candidate_score(
    features: CandidateFeatures,
    config: HeuristicViaDetectorConfig,
    allowed_diameters: list[int],
    line_likeness: float,
) -> float:
    """Combine normalized features; all calibration constants live in config."""

    return _candidate_score_metrics(features, config, allowed_diameters, line_likeness)["final_score"]


def _scale01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _debug_viz(
    gray: np.ndarray,
    corrected: np.ndarray,
    mbright: np.ndarray,
    mdark: np.ndarray,
    acc: list[ViaDetection],
    below: list[ViaDetection],
    hard: list[ViaDetection],
    d: int,
) -> dict[str, np.ndarray]:
    h, w = gray.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    g = (0, 200, 0)
    yl = (0, 220, 255)
    rd = (0, 0, 255)
    for v in acc:
        cv2.drawMarker(
            out,
            (int(round(v.x)), int(round(v.y))),
            g,
            markerType=cv2.MARKER_CROSS,
            markerSize=int(max(5, v.diameter_estimate + 1)),
            thickness=1,
        )
    for v in below:
        cv2.drawMarker(
            out,
            (int(round(v.x)), int(round(v.y))),
            yl,
            markerType=cv2.MARKER_SQUARE,
            markerSize=int(max(5, d)),
            thickness=1,
        )
    for v in hard:
        if v.reject_reason and "low_contrast" not in str(v.reject_reason):
            cv2.drawMarker(
                out,
                (int(round(v.x)), int(round(v.y))),
                rd,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=int(max(5, d)),
                thickness=1,
            )
    return {
        "source_gray": base,
        "background_corrected": cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR),
        "local_max_bright": cv2.cvtColor(mbright, cv2.COLOR_GRAY2BGR) if mbright is not None else base,
        "local_max_dark": cv2.cvtColor(mdark, cv2.COLOR_GRAY2BGR) if mdark is not None else base,
        "overlay": cv2.addWeighted(base, 0.55, out, 0.7, 0) if out.size else base,
    }
