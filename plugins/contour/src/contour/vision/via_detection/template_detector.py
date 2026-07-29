"""Template-based via detection: multi-template matchTemplate + NMS (not the main default)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TemplateViaDetectorConfig
from .result import DetectionResult, ViaDetection

_MAX_BELOW_THRESHOLD_PER_TEMPLATE_SCALE = 2_000
_MAX_BELOW_THRESHOLD_RESULTS = 10_000


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
            threshold = _template_threshold(config, template_index) / 100.0
            _collect_peaks(
                res,
                t,
                threshold,
                all_dets,
                template_index=template_index,
                below_threshold_limit=_MAX_BELOW_THRESHOLD_PER_TEMPLATE_SCALE,
            )
    return all_dets, (h, w)


def score_vias_template_raw(
    raw_matches: list[TemplateRawMatch], image_shape: tuple[int, int], config: TemplateViaDetectorConfig
) -> DetectionResult:
    h, w = image_shape
    base = dict(config.snapshot())
    eligible: list[tuple[TemplateRawMatch, ViaDetection]] = []
    below: list[ViaDetection] = []
    below_threshold_count = 0
    for raw in sorted(raw_matches, key=lambda item: item.score, reverse=True):
        detection = _raw_match_detection(raw, config)
        if detection.score >= _template_threshold(config, raw.template_index):
            eligible.append((raw, detection))
            continue
        below_threshold_count += 1
        if len(below) < _MAX_BELOW_THRESHOLD_RESULTS:
            below.append(detection)
    kept = _suppress_duplicate_matches(eligible)
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
        parameters_snapshot={
            **base,
            "raw_matches": len(raw_matches),
            "below_threshold_count": below_threshold_count,
            "below_threshold_debug_count": len(below),
        },
    )


def _template_threshold(config: TemplateViaDetectorConfig, template_index: int) -> float:
    value = (
        config.min_correlations[template_index]
        if template_index < len(config.min_correlations)
        else config.min_correlation
    )
    return max(0.0, min(1.0, float(value))) * 100.0


def _raw_match_detection(
    raw: TemplateRawMatch,
    config: TemplateViaDetectorConfig,
) -> ViaDetection:
    return ViaDetection(
        raw.x,
        raw.y,
        raw.bbox,
        raw.score,
        float(config.output_diameters[raw.template_index])
        if raw.template_index < len(config.output_diameters)
        else raw.diameter_estimate,
        raw.score * 0.32,
        raw.score * 0.20,
        0.5,
        float(max(raw.bbox[2], raw.bbox[3]) / (min(raw.bbox[2], raw.bbox[3]) + 1e-6)),
        "template",
        None,
        raw.template_index,
    )


def _suppress_duplicate_matches(
    eligible: list[tuple[TemplateRawMatch, ViaDetection]],
) -> list[tuple[TemplateRawMatch, ViaDetection]]:
    """Apply the original variable-radius NMS using a uniform spatial grid."""

    if not eligible:
        return []
    radii = [max(raw.bbox[2], raw.bbox[3]) + 2 for raw, _detection in eligible]
    cell_size = max(1, max(radii))
    kept: list[tuple[TemplateRawMatch, ViaDetection]] = []
    kept_radii: list[int] = []
    grid: dict[tuple[int, int], list[int]] = {}

    for (raw, detection), distance in zip(eligible, radii, strict=True):
        cell_x = int(detection.x // cell_size)
        cell_y = int(detection.y // cell_size)
        duplicate = False
        for grid_y in range(cell_y - 1, cell_y + 2):
            for grid_x in range(cell_x - 1, cell_x + 2):
                for kept_index in grid.get((grid_x, grid_y), ()):
                    other = kept[kept_index][1]
                    threshold = max(distance, kept_radii[kept_index])
                    if (
                        (detection.x - other.x) ** 2
                        + (detection.y - other.y) ** 2
                        <= float(threshold * threshold)
                    ):
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        kept_index = len(kept)
        kept.append((raw, detection))
        kept_radii.append(distance)
        grid.setdefault((cell_x, cell_y), []).append(kept_index)
    return kept


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
    below_threshold_limit: int = 0,
) -> None:
    if res.size == 0:
        return
    th, tw = tmpl.shape[:2]
    # Non-max: dilate
    k = max(3, min(th, tw) // 2 | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(res, ker)
    loc = res == dil
    ys, xs = np.where(loc)
    if ys.size == 0:
        return
    values = res[ys, xs]
    accepted_indices = np.flatnonzero(values >= float(thr))
    below_indices = np.flatnonzero(values < float(thr))
    limit = max(0, int(below_threshold_limit))
    if below_indices.size > limit:
        if limit == 0:
            below_indices = below_indices[:0]
        else:
            relative = np.argpartition(values[below_indices], -limit)[-limit:]
            below_indices = below_indices[relative]
    selected_indices = np.concatenate((accepted_indices, below_indices))
    for index in selected_indices:
        y, x = ys[index], xs[index]
        v = float(res[y, x])
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
