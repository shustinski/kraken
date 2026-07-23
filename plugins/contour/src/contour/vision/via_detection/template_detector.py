"""Template-based via detection: multi-template matchTemplate + NMS (not the main default)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TemplateViaDetectorConfig
from .result import DetectionResult, ViaDetection


@dataclass(frozen=True, slots=True)
class TemplateRawMatch:
    x: float
    y: float
    bbox: tuple[int, int, int, int]
    score: float
    diameter_estimate: float
    template_index: int = 0


def detect_vias_template(image: np.ndarray, config: TemplateViaDetectorConfig) -> DetectionResult:
    raw, shape = detect_vias_template_raw(image, config)
    return score_vias_template_raw(raw, shape, config)


def detect_vias_template_raw(
    image: np.ndarray, config: TemplateViaDetectorConfig
) -> tuple[list[TemplateRawMatch], tuple[int, int]]:
    g = _to_gray_u8(image)
    h, w = g.shape[:2]
    if h < 3 or w < 3 or not config.templates:
        return [], (h, w)

    method = cv2.TM_CCOEFF_NORMED if config.use_ccoeff_normed else cv2.TM_CCORR_NORMED
    all_dets: list[TemplateRawMatch] = []
    scales = _iter_scales(float(config.scale_min), float(config.scale_max), float(config.scale_step))

    for template_index, tmpl in enumerate(config.templates):
        t0 = np.asarray(tmpl, dtype=np.uint8)
        if t0.ndim > 2:
            t0 = cv2.cvtColor(t0, cv2.COLOR_BGR2GRAY)
        th0, tw0 = t0.shape[:2]
        if th0 < 2 or tw0 < 2 or th0 >= h or tw0 >= w:
            continue
        for sc in scales:
            th, tw = max(2, round(th0 * sc)), max(2, round(tw0 * sc))
            if th >= h or tw >= w:
                continue
            t = cv2.resize(t0, (tw, th), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(g, t, method)
            floor = 0.0
            _collect_peaks(res, t, floor, all_dets, template_index=template_index)
    return all_dets, (h, w)


def score_vias_template_raw(
    raw_matches: list[TemplateRawMatch], image_shape: tuple[int, int], config: TemplateViaDetectorConfig
) -> DetectionResult:
    h, w = image_shape
    base = dict(config.snapshot())
    all_dets = [
        ViaDetection(
            d.x,
            d.y,
            d.bbox,
            d.score,
            float(config.output_diameters[d.template_index])
            if d.template_index < len(config.output_diameters)
            else d.diameter_estimate,
            d.score * 0.32,
            d.score * 0.20,
            0.5,
            float(max(d.bbox[2], d.bbox[3]) / (min(d.bbox[2], d.bbox[3]) + 1e-6)),
            "template",
            None,
            d.template_index,
        )
        for d in raw_matches
    ]
    indexed_dets = list(zip(raw_matches, all_dets, strict=True))
    indexed_dets.sort(key=lambda pair: pair[1].score, reverse=True)

    def threshold(raw: TemplateRawMatch) -> float:
        value = (
            config.min_correlations[raw.template_index]
            if raw.template_index < len(config.min_correlations)
            else config.min_correlation
        )
        return max(0.0, min(1.0, float(value))) * 100.0

    eligible = [(raw, detection) for raw, detection in indexed_dets if detection.score >= threshold(raw)]
    below = [detection for raw, detection in indexed_dets if detection.score < threshold(raw)]
    kept: list[tuple[TemplateRawMatch, ViaDetection]] = []
    for raw, detection in eligible:
        # Duplicate suppression is automatic: the matched template span plus
        # two pixels. For two different templates the larger distance wins.
        distance = max(raw.bbox[2], raw.bbox[3]) + 2
        duplicate = any(
            (detection.x - other.x) ** 2 + (detection.y - other.y) ** 2
            <= float(max(distance, max(other_raw.bbox[2], other_raw.bbox[3]) + 2) ** 2)
            for other_raw, other in kept
        )
        if not duplicate:
            kept.append((raw, detection))
    acc = [detection for _raw, detection in kept]

    dbg = {
        "source_gray": np.zeros((h, w, 3), np.uint8),
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
    res: np.ndarray,
    tmpl: np.ndarray,
    thr: float,
    out: list[TemplateRawMatch],
    *,
    template_index: int,
) -> None:
    if res.size == 0:
        return
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
            TemplateRawMatch(
                x=cx,
                y=cy,
                bbox=(int(x), int(y), int(tw), int(th)),
                score=v * 100.0,
                diameter_estimate=float((tw + th) * 0.5),
                template_index=template_index,
            )
        )
