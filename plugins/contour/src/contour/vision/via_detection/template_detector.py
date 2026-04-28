"""Template-based via detection: multi-template matchTemplate + NMS (not the main default)."""

from __future__ import annotations

import numpy as np
import cv2

from .config import TemplateViaDetectorConfig
from .result import DetectionResult, ViaDetection


def detect_vias_template(image: np.ndarray, config: TemplateViaDetectorConfig) -> DetectionResult:
    g = _to_gray_u8(image)
    h, w = g.shape[:2]
    base = dict(config.snapshot())
    if h < 3 or w < 3 or not config.templates:
        return DetectionResult(method="template", accepted=[], parameters_snapshot=base)

    method = cv2.TM_CCOEFF_NORMED if config.use_ccoeff_normed else cv2.TM_CCORR_NORMED
    all_dets: list[ViaDetection] = []
    scales = _iter_scales(float(config.scale_min), float(config.scale_max), float(config.scale_step))

    for tmpl in config.templates:
        t0 = np.asarray(tmpl, dtype=np.uint8)
        if t0.ndim > 2:
            t0 = cv2.cvtColor(t0, cv2.COLOR_BGR2GRAY)
        th0, tw0 = t0.shape[:2]
        if th0 < 2 or tw0 < 2 or th0 >= h or tw0 >= w:
            continue
        for sc in scales:
            th, tw = max(2, int(round(th0 * sc))), max(2, int(round(tw0 * sc)))
            if th >= h or tw >= w:
                continue
            t = cv2.resize(t0, (tw, th), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(g, t, method)
            floor = max(0.1, float(config.min_correlation) * 0.35)
            _collect_peaks(res, t, floor, all_dets)

    nmsd = max(0, int(config.nms_distance))
    d2 = float(nmsd * nmsd) if nmsd else 0.0
    all_dets.sort(key=lambda d: d.score, reverse=True)
    kept: list[ViaDetection] = []
    for d in all_dets:
        if d2 == 0:
            kept.append(d)
        elif not any((d.x - k.x) ** 2 + (d.y - k.y) ** 2 <= d2 for k in kept):
            kept.append(d)

    thr = float(config.min_correlation) * 100.0
    acc = [d for d in kept if d.score >= thr]
    below = [d for d in kept if d.score < thr]

    dbg = {
        "source_gray": cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
        "template_count": np.zeros((h, w, 3), np.uint8),
    }
    return DetectionResult(
        method="template",
        accepted=acc,
        rejected=[],
        below_threshold=below,
        debug_images=dbg,
        parameters_snapshot={**base, "raw_matches": len(all_dets)},
    )


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[:, :, 0].astype(np.uint8, copy=False)


def _iter_scales(smin: float, smax: float, step: float) -> list[float]:
    if smax < smin:
        smin, smax = smax, smin
    if step <= 0:
        return [1.0]
    out: list[float] = []
    x = smin
    while x <= smax + 1e-6:
        out.append(float(round(x, 4)))
        x += step
    return out or [1.0]


def _collect_peaks(
    res: np.ndarray, tmpl: np.ndarray, thr: float, out: list[ViaDetection]
) -> None:
    if res.size == 0:
        return
    rh, rw = res.shape[:2]
    th, tw = tmpl.shape[:2]
    # Non-max: dilate
    k = max(3, min(th, tw) // 2 | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(res, ker)
    loc = (res == dil) & (res >= float(thr))
    ys, xs = np.where(loc)
    for y, x in zip(ys, xs, strict=False):
        v = float(res[y, x])
        if v < thr:
            continue
        cx = float(x) + float(tw) * 0.5
        cy = float(y) + float(th) * 0.5
        out.append(
            ViaDetection(
                cx,
                cy,
                (int(x), int(y), int(tw), int(th)),
                v * 100.0,
                float((tw + th) * 0.5),
                v * 32.0,
                v * 20.0,
                0.5,
                float(max(tw, th) / (min(tw, th) + 1e-6)),
                "template",
                None,
            )
        )
