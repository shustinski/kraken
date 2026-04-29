"""Heuristic via detection: local extrema + local component + structural scoring (no top-hat/DoG/ML)."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .config import HeuristicViaDetectorConfig, ViaPolarity
from .result import DetectionResult, ViaDetection


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
    g = _to_gray_u8(image)
    h, w = g.shape[:2]
    snap0 = dict(config.snapshot())
    if h < 3 or w < 3:
        return DetectionResult(method="heuristic", accepted=[], parameters_snapshot=snap0)

    allowed = config.allowed_diameters()
    if not allowed:
        return DetectionResult(method="heuristic", accepted=[], parameters_snapshot=snap0)
    d_min, d_max = min(allowed), max(allowed)

    sens = _sensitivity_map(config.sensitivity)
    g_pre = _preprocess_denoise(g, config)
    bg = cv2.GaussianBlur(g_pre, (0, 0), float(config.background_sigma))
    corr = g_pre.astype(np.float32) - bg.astype(np.float32)
    corr_u8 = _normalize01_to_u8(corr)

    bright_map = corr_u8
    dark_map = 255 - corr_u8

    ksize = int(max(3, 2 * int(round(0.5 * d_max)) + 1))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    seeds_b, mask_b = _local_extrema_seeds(bright_map, ker, sens["percentile"], float(config.min_peak_grey))
    seeds_d, mask_d = _local_extrema_seeds(dark_map, ker, sens["percentile"], float(config.min_peak_grey))

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
        hyps = [
            str(ViaPolarity.BRIGHT),
            str(ViaPolarity.DARK),
            str(ViaPolarity.RING_LIGHT_RING),
            str(ViaPolarity.RING_DARK_RING),
        ]
    elif polar in (str(ViaPolarity.RING_LIGHT_RING), str(ViaPolarity.RING_DARK_RING), ViaPolarity.RING_LIGHT_RING, ViaPolarity.RING_DARK_RING):
        hyps = [polar] if not isinstance(polar, ViaPolarity) else [str(polar)]
    else:
        hyps = [polar]

    dets: list[ViaDetection] = []
    patch_scale = max(1.0, float(config.analysis_window_scale))
    min_ps = int(config.min_analyze_size)

    for cy, cx in raw_seeds:
        best: ViaDetection | None = None
        for d_est in allowed:
            psize = int(max(min_ps, round(patch_scale * float(d_est))))
            half = psize // 2
            y0, y1 = max(0, cy - half), min(h, cy + half + 1)
            x0, x1 = max(0, cx - half), min(w, cx + half + 1)
            patch = g_pre[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            pcx, pcy = cx - x0, cy - y0
            for hyp in hyps:
                if hyp in {
                    str(ViaPolarity.AUTO),
                    "auto",
                }:
                    continue
                det = _score_one(
                    patch,
                    pcx,
                    pcy,
                    d_est,
                    (x0, y0),
                    hyp,
                    config,
                )
                if det is None:
                    continue
                if _det_better(det, best):
                    best = det
        if best is not None:
            dets.append(best)

    scored_count = len(dets)
    dets = [d for d in dets if d.reject_reason is None or (d.reject_reason and "hard" in d.reject_reason)]
    dets = _dedupe_by_score(dets, min_dist=1.0)
    dets.sort(key=lambda d: d.score, reverse=True)
    after = _nms_simple(dets, max(0, int(config.nms_distance)))

    accepted: list[ViaDetection] = [
        d
        for d in after
        if d.reject_reason is None and d.score >= float(config.min_final_score)
    ]
    below: list[ViaDetection] = [
        d
        for d in after
        if d.reject_reason is None and 0 < d.score < float(config.min_final_score)
    ]
    hard = [d for d in after if d.reject_reason and "hard" in d.reject_reason]

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
            "accepted_count": len(accepted),
            "below_threshold_count": len(below),
            "hard_rejected_count": len(hard),
            "sensitivity": sens,
        },
    )


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


def _sensitivity_map(s: str) -> dict[str, float]:
    t = (s or "medium").strip().lower()
    if t in {"low", "низ", "низкая"}:
        return {"percentile": 99.2}
    if t in {"high", "выс", "высокая"}:
        return {"percentile": 96.8}
    return {"percentile": 98.3}


def _local_extrema_seeds(
    response: np.ndarray, kernel: np.ndarray, pctl: float, min_peak: float
) -> tuple[list[tuple[int, int]], np.ndarray]:
    dil = cv2.dilate(response, kernel)
    lm = (response == dil) & (response > 0)
    th = max(float(min_peak), float(np.percentile(response, pctl)))
    lm = lm & (response >= th)
    pts: list[tuple[int, int]] = []
    ys, xs = np.where(lm)
    for y, x in zip(ys, xs, strict=False):
        pts.append((int(y), int(x)))
    m = (lm.astype(np.uint8) * 255) if hasattr(lm, "astype") else np.zeros_like(response, dtype=np.uint8)
    return pts, m


def _merge_seeds(allp: list[tuple[int, int]], min_dist: int, h: int, w: int) -> list[tuple[int, int]]:
    if not allp:
        return []
    d2 = float(max(1, min_dist) ** 2)
    allp = sorted(set(allp), key=lambda p: p[0] * 1_000_000 + p[1])
    out: list[tuple[int, int]] = []
    for p in allp:
        if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < d2 for q in out):
            continue
        out.append(p)
    return out


def _spread_points(pts: list[tuple[int, int]], min_dist: int, h: int, w: int) -> list[tuple[int, int]]:
    if not pts:
        return []
    d2 = float(max(1, min_dist) ** 2)
    pts = sorted(pts, key=lambda p: p[0] * 1_000_000 + p[1])
    out: list[tuple[int, int]] = []
    for p in pts:
        if any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < d2 for q in out):
            continue
        out.append(p)
    return out


def _dedupe_by_score(dets: list[ViaDetection], min_dist: float) -> list[ViaDetection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    d2 = float(max(0.5, min_dist) ** 2)
    kept: list[ViaDetection] = []
    for d in dets:
        if d.reject_reason and "hard" in d.reject_reason:
            kept.append(d)
            continue
        if any((d.x - k.x) ** 2 + (d.y - k.y) ** 2 < d2 for k in kept if not (k.reject_reason and "hard" in k.reject_reason)):
            continue
        kept.append(d)
    return kept


def _nms_simple(dets: list[ViaDetection], dist: int) -> list[ViaDetection]:
    hard = [d for d in dets if d.reject_reason and "hard" in str(d.reject_reason)]
    soft = [d for d in dets if not (d.reject_reason and "hard" in str(d.reject_reason))]
    soft.sort(key=lambda x: x.score, reverse=True)
    d2 = float(max(0, int(dist)) ** 2) if dist > 0 else 0.0
    keep: list[ViaDetection] = []
    for d in soft:
        if d2 == 0.0:
            keep.append(d)
        elif not any((d.x - o.x) ** 2 + (d.y - o.y) ** 2 <= d2 for o in keep):
            keep.append(d)
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
        yy, xx = np.indices((ph, pw), dtype=np.float32)
        return float((xx * wts).sum() / sw), float((yy * wts).sum() / sw)
    mm = cv2.moments((mask_bool.astype(np.uint8) * 255), binaryImage=True)
    if mm.get("m00", 0) and float(mm["m00"]) > 1e-6:
        return float(mm["m10"] / mm["m00"]), float(mm["m01"] / mm["m00"])
    return float(seed_x), float(seed_y)


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


def _score_one(
    patch: np.ndarray,
    pcx: int,
    pcy: int,
    d_est: int,
    offset: tuple[int, int],
    hyp: str,
    config: HeuristicViaDetectorConfig,
) -> ViaDetection | None:
    h, w = patch.shape[:2]
    if pcx < 0 or pcy < 0 or pcx >= w or pcy >= h:
        return None
    ph, pw = h, w
    center_m, inner_m, outer_m = _annulus_masks((ph, pw), pcx, pcy, float(d_est))
    gpatch = patch.astype(np.float32)
    contrast = _contrast_for_polarity(gpatch, center_m, inner_m, outer_m, hyp)
    if contrast < float(config.min_center_contrast):
        return ViaDetection(
            float(offset[0] + pcx),
            float(offset[1] + pcy),
            (0, 0, 0, 0),
            0.0,
            float(d_est),
            float(contrast),
            0.0,
            0.0,
            0.0,
            hyp,
            "hard:low_contrast",
        )
    med = float(np.median(gpatch))
    pr = min(4, max(1, max(h, w) // 6))
    ny0, ny1 = max(0, pcy - pr), min(h, pcy + pr + 1)
    nx0, nx1 = max(0, pcx - pr), min(w, pcx + pr + 1)
    nh = gpatch[ny0:ny1, nx0:nx1].astype(np.float32).ravel()
    prom = float(np.max(np.abs(nh - med))) if nh.size else 0.0
    if prom < float(config.min_peak_prominence):
        return None

    p = float(config.local_binarize_percentile)
    delta = max(float(config.min_center_contrast), abs(float(contrast)) * 0.30, 2.0)
    if hyp in (str(ViaPolarity.DARK), "dark"):
        thr2 = min(float(np.percentile(gpatch, max(1.0, 100.0 - p))), med - delta)
        binm = (gpatch <= thr2).astype(np.uint8) * 255
    elif hyp in (str(ViaPolarity.RING_DARK_RING), "ring_dark_ring") and gpatch[pcy, pcx] < med:
        thr2 = min(float(np.percentile(gpatch, max(1.0, 100.0 - p))), med - delta)
        binm = (gpatch <= thr2).astype(np.uint8) * 255
    else:
        thr = max(float(np.percentile(gpatch, p)), med + delta)
        binm = (gpatch >= thr).astype(np.uint8) * 255
    nlab, lab, stat, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    lab_at = int(lab[pcy, pcx])
    if lab_at <= 0:
        return ViaDetection(
            float(offset[0] + pcx),
            float(offset[1] + pcy),
            (0, 0, 0, 0),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            0.0,
            0.0,
            hyp,
            "hard:no_component",
        )
    area = float(stat[lab_at, cv2.CC_STAT_AREA])
    if area < 1.0:
        return None
    comp = (lab == lab_at).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt0 = max(cnts, key=cv2.contourArea)
    eq_diam = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
    tol = config.effective_size_tolerance()
    re = max(float(d_est), 1.0)
    if abs(eq_diam - float(d_est)) / re > tol:
        return ViaDetection(
            float(offset[0] + pcx),
            float(offset[1] + pcy),
            (0, 0, 0, 0),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            0.0,
            0.0,
            hyp,
            f"hard:size_mismatch(eq={eq_diam:.1f},d={d_est})",
        )
    fcx, fcy = _refine_center_xy(gpatch, (lab == lab_at), seed_x=float(pcx), seed_y=float(pcy), med=med, hyp=hyp)
    drift = math.hypot(fcx - float(pcx), fcy - float(pcy))
    max_drift = float(config.max_center_drift_ratio) * re
    if drift > max_drift:
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            (0, 0, 0, 0),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            0.0,
            0.0,
            hyp,
            f"hard:center_drift({drift:.1f}>{max_drift:.1f})",
        )
    r_rect = cv2.minAreaRect(cnt0)
    w_r, h_r = float(r_rect[1][0]), float(r_rect[1][1])
    if w_r < 1e-3 or h_r < 1e-3:
        aspect = 1.0
    else:
        aspect = max(w_r, h_r) / (min(w_r, h_r) + 1e-6)
    r_expect = d_est * 0.5
    perim = float(cv2.arcLength(cnt0, True)) + 1e-3
    circ2 = 4.0 * math.pi * max(area, 1.0) / (perim * perim) if perim > 0 else 0.0
    fill = 4.0 * area / (w_r * h_r + 1e-6) if w_r * h_r > 1e-6 else 0.0
    compact2 = 0.5 * min(1.0, min(fill, 1.0)) + 0.5 * min(1.0, area / (math.pi * r_expect**2 + 1e-3))
    if compact2 < float(config.min_compactness):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            (0, 0, 0, 0),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            "hard:compact",
        )
    if aspect > float(config.max_elongation):
        return ViaDetection(
            float(offset[0] + fcx),
            float(offset[1] + fcy),
            (0, 0, 0, 0),
            0.0,
            float(d_est),
            float(contrast),
            float(prom),
            float(compact2),
            float(aspect),
            hyp,
            "hard:elongation",
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
    line_n = min(1.0, el / (float(config.max_elongation) + 0.1))

    sc_c = _scale01(contrast, 3.0, 20.0)
    sc_p = _scale01(prom, 2.0, 25.0)
    d_lo = float(min(config.allowed_diameters()))
    d_hi = float(max(config.allowed_diameters()))
    sc_z = 1.0 - min(1.0, abs(d_est - 0.5 * (d_lo + d_hi)) / (d_hi - d_lo + 1.0) * 0.4)
    sc_k = min(1.0, max(0.0, float(compact2)))
    sc_r = min(1.0, max(0.0, min(circ2, 1.0)))
    sc_b = 1.0 - min(1.0, border_n * 1.2)

    W = config
    raw = (
        W.w_contrast * sc_c
        + W.w_prominence * sc_p
        + W.w_size * sc_z
        + W.w_compact * sc_k
        + W.w_round * sc_r
        + W.w_balance * sc_b
        - W.w_line * line_n * float(W.line_penalty_scale)
        - W.w_border * border_n * float(W.border_penalty_scale)
    )
    final = max(0.0, min(100.0, raw))

    gx = float(offset[0]) + fcx
    gy = float(offset[1]) + fcy
    half = float(d_est) * 0.5
    ox = int(round(gx - half))
    oy = int(round(gy - half))
    bbox = (ox, oy, int(d_est), int(d_est))
    return ViaDetection(
        gx,
        gy,
        bbox,
        float(final),
        float(d_est),
        float(contrast),
        float(prom),
        float(compact2),
        float(aspect),
        hyp,
        None,
    )


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
