"""Primary SEM via detector.

The detector is not a contour/blob-circularity wrapper. It generates candidates
from several cues, then scores each local ROI:

* polarity-aware multi-scale top-hat / black-hat response;
* signed LoG spot response;
* annular edge response on a Scharr gradient map;
* radial contrast between the center and surrounding annulus;
* angular coverage of the edge ring;
* connected-component isolation to suppress long trace segments.

Legacy template matching remains outside this module and is merged by
``via.orchestrator`` as a separate strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..io_normalize import to_gray_u8
from ..preprocessing import flatten_illumination
from ..schemas import ViaHit


@dataclass(frozen=True, slots=True)
class ViaPolarityScan:
    bright: bool
    low: int = 0
    high: int = 255


@dataclass(frozen=True, slots=True)
class SemPrimaryViaConfig:
    """Internal detector tuning. The UI should expose presets, not every field."""

    expected_diameters: tuple[int, ...] = ()
    min_width: float = 2.0
    max_width: float = 48.0
    min_height: float = 2.0
    max_height: float = 48.0
    scans: tuple[ViaPolarityScan, ...] = (ViaPolarityScan(bright=True), ViaPolarityScan(bright=False))
    min_score: float = 0.35
    min_contrast: float = 10.0
    min_edge_coverage: float = 0.25
    line_suppression: float = 0.65
    max_candidates: int = 1500

    @classmethod
    def from_legacy_settings(cls, settings: Any) -> SemPrimaryViaConfig:
        expected = _expected_diameters(settings)
        min_width, max_width, min_height, max_height = _size_limits(settings, expected)
        scans = tuple(_polarity_scans(settings))
        return cls(
            expected_diameters=tuple(expected),
            min_width=float(min_width),
            max_width=float(max_width),
            min_height=float(min_height),
            max_height=float(max_height),
            scans=scans,
            min_score=max(0.0, min(1.0, float(getattr(settings, "via_min_score", 0.35)))),
            min_contrast=max(0.0, float(getattr(settings, "via_min_contrast", 10.0))),
            min_edge_coverage=max(0.0, min(1.0, float(getattr(settings, "via_min_edge_coverage", 0.25)))),
            line_suppression=max(0.0, min(1.0, float(getattr(settings, "via_spot_line_suppression", 0.65)))),
        )


@dataclass(frozen=True, slots=True)
class _ViaCandidate:
    center_x: float
    center_y: float
    width: float
    height: float
    score: float
    bright: bool
    contrast: float
    edge_strength: float
    annulus_coverage: float
    response_score: float
    isolation_score: float
    source: str = "sem_primary"

    def to_hit(self) -> ViaHit:
        return ViaHit(
            center_x=float(self.center_x),
            center_y=float(self.center_y),
            width=float(self.width),
            height=float(self.height),
            score=float(self.score),
            strategy=self.source,
            contrast=float(self.contrast),
            edge_strength=float(self.edge_strength),
            annulus_coverage=float(self.annulus_coverage),
            extra={
                "response_score": float(self.response_score),
                "isolation_score": float(self.isolation_score),
                "polarity": "bright" if self.bright else "dark",
            },
        )


@dataclass(slots=True)
class SemPrimaryViaDetector:
    config: SemPrimaryViaConfig

    def detect(self, image: Any) -> list[ViaHit]:
        gray = to_gray_u8(image)
        if gray.size == 0:
            return []
        prepared = _prepare(gray)
        gradient = _gradient(prepared)
        radii = _candidate_radii(self.config)
        if not radii:
            return []

        candidates: list[_ViaCandidate] = []
        for scan in self.config.scans:
            gate = _intensity_gate(gray, prepared, scan)
            response = _candidate_response(prepared, gradient, radii, scan.bright, self.config.line_suppression)
            if cv2.countNonZero(gate) > 0:
                response = cv2.bitwise_and(response, gate)
            peaks = _extract_peaks(response, min_distance=max(2, int(np.median(radii) * 1.35)))
            isolation = _build_isolation(gate, gray, scan.bright, max_radius=max(radii))
            for cy, cx in peaks[: self.config.max_candidates]:
                candidate = _verify_candidate(
                    gray=gray,
                    prepared=prepared,
                    gradient=gradient,
                    response=response,
                    cx=int(cx),
                    cy=int(cy),
                    radii=radii,
                    bright=scan.bright,
                    isolation=isolation,
                )
                if candidate is not None and self._accepts(candidate):
                    candidates.append(candidate)

        kept = _nms(candidates, iou_threshold=0.35)
        kept.sort(key=lambda item: (item.center_y, item.center_x))
        return [candidate.to_hit() for candidate in kept]

    def _accepts(self, candidate: _ViaCandidate) -> bool:
        if candidate.width < self.config.min_width or candidate.width > self.config.max_width:
            return False
        if candidate.height < self.config.min_height or candidate.height > self.config.max_height:
            return False
        if candidate.contrast < self.config.min_contrast:
            return False
        if candidate.annulus_coverage < self.config.min_edge_coverage:
            return False
        return candidate.score >= self.config.min_score


def sem_primary_hits(
    image: Any,
    settings: Any,
    log: list[str],
) -> list[ViaHit]:
    """Compatibility entrypoint used by the composite orchestrator."""

    config = SemPrimaryViaConfig.from_legacy_settings(settings)
    log.append(
        "sem_primary: multi-cue via detector "
        f"diameters={list(config.expected_diameters) or 'auto'} "
        f"scans={[('bright' if s.bright else 'dark', s.low, s.high) for s in config.scans]}"
    )
    return SemPrimaryViaDetector(config).detect(image)


def _prepare(gray: np.ndarray) -> np.ndarray:
    g = to_gray_u8(gray)
    if min(g.shape[:2]) >= 32:
        g = flatten_illumination(g, 0.04)
    tile = max(4, min(12, _odd(min(g.shape[:2]) / 24.0, minimum=5)))
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile)).apply(g)
    if min(g.shape[:2]) >= 8:
        g = cv2.fastNlMeansDenoising(g, h=5, templateWindowSize=7, searchWindowSize=15)
    else:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    return g


def _gradient(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(to_gray_u8(gray), (3, 3), 0)
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return _normalize_float(mag)


def _candidate_response(
    gray: np.ndarray,
    gradient: np.ndarray,
    radii: list[int],
    bright: bool,
    line_suppression: float,
) -> np.ndarray:
    diameter = max(3, int(np.median(radii) * 2 + 1))
    morph = _multiscale_morph_response(gray, diameter, bright=bright, line_suppression=line_suppression)
    log = _log_response(gray, radii, bright=bright)
    ring = _ring_response(gradient, radii)
    spot = cv2.max(morph, log)
    return cv2.addWeighted(spot, 0.68, ring, 0.32, 0)


def _multiscale_morph_response(
    gray: np.ndarray,
    expected_diameter: int,
    *,
    bright: bool,
    line_suppression: float,
) -> np.ndarray:
    data = to_gray_u8(gray)
    operation = cv2.MORPH_TOPHAT if bright else cv2.MORPH_BLACKHAT
    response = np.zeros_like(data)
    for scale in (1.6, 2.4, 3.2):
        k = _odd(expected_diameter * scale, minimum=5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        response = cv2.max(response, cv2.morphologyEx(data, operation, kernel))
    if line_suppression > 0.0:
        line_len = _odd(max(9, expected_diameter * 4.0), minimum=9)
        line_w = _odd(max(3, expected_diameter * 0.45), minimum=3)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, line_w))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, line_len))
        lines = cv2.max(
            cv2.morphologyEx(response, cv2.MORPH_OPEN, h_kernel), cv2.morphologyEx(response, cv2.MORPH_OPEN, v_kernel)
        )
        response = cv2.subtract(response, cv2.convertScaleAbs(lines, alpha=line_suppression))
    return response


def _log_response(gray: np.ndarray, radii: list[int], *, bright: bool) -> np.ndarray:
    data = to_gray_u8(gray).astype(np.float32)
    acc = np.zeros_like(data, dtype=np.float32)
    for radius in radii:
        sigma = max(0.65, float(radius) * 0.55)
        blurred = cv2.GaussianBlur(data, (0, 0), sigma)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        signed = -lap if bright else lap
        local = np.maximum(signed * float(sigma * sigma), 0.0)
        values = local[local > 0]
        if values.size == 0:
            continue
        scale = max(35.0, float(np.percentile(values, 99.5)))
        acc = np.maximum(acc, np.clip(local * (255.0 / scale), 0.0, 255.0))
    return acc.astype(np.uint8)


def _ring_response(gradient: np.ndarray, radii: list[int]) -> np.ndarray:
    if gradient.size == 0:
        return np.zeros_like(gradient, dtype=np.uint8)
    search = gradient.astype(np.float32) / 255.0
    acc = np.zeros_like(search, dtype=np.float32)
    for radius in radii:
        radius = max(2, int(radius))
        size = radius * 2 + 5
        if gradient.shape[0] < size or gradient.shape[1] < size:
            continue
        center = size // 2
        template = np.zeros((size, size), dtype=np.float32)
        cv2.circle(template, (center, center), radius, 1.0, max(1, round(radius * 0.22)), lineType=cv2.LINE_AA)
        scores = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        scores = np.nan_to_num(scores, copy=False)
        placed = np.zeros_like(acc)
        ph, pw = scores.shape
        placed[center : center + ph, center : center + pw] = scores
        acc = np.maximum(acc, placed)
    return np.clip(acc * 255.0, 0.0, 255.0).astype(np.uint8)


def _verify_candidate(
    *,
    gray: np.ndarray,
    prepared: np.ndarray,
    gradient: np.ndarray,
    response: np.ndarray,
    cx: int,
    cy: int,
    radii: list[int],
    bright: bool,
    isolation: tuple[np.ndarray, np.ndarray] | None,
) -> _ViaCandidate | None:
    best: _ViaCandidate | None = None
    best_score = -1.0
    for radius in radii:
        radius = max(2, int(radius))
        contrast = max(
            _radial_contrast(gray, cx, cy, radius, bright=bright),
            _radial_contrast(prepared, cx, cy, radius, bright=bright),
        )
        edge_strength, coverage = _edge_ring_metrics(gradient, cx, cy, radius)
        response_score = (
            float(response[cy, cx]) / 255.0 if 0 <= cy < response.shape[0] and 0 <= cx < response.shape[1] else 0.0
        )
        isolation_score = _isolation_score(isolation, cx, cy, radius)
        score = (
            min(1.0, contrast / 80.0) * 0.34
            + min(1.0, edge_strength / 80.0) * 0.22
            + coverage * 0.24
            + response_score * 0.12
            + isolation_score * 0.08
        )
        if score > best_score:
            refined_x, refined_y = _refine_center(prepared, cx, cy, radius, bright=bright)
            best_score = score
            best = _ViaCandidate(
                center_x=refined_x,
                center_y=refined_y,
                width=float(radius * 2 + 1),
                height=float(radius * 2 + 1),
                score=float(score),
                bright=bright,
                contrast=float(contrast),
                edge_strength=float(edge_strength),
                annulus_coverage=float(coverage),
                response_score=float(response_score),
                isolation_score=float(isolation_score),
            )
    return best


def _radial_contrast(gray: np.ndarray, cx: int, cy: int, radius: int, *, bright: bool) -> float:
    radius = max(2, int(radius))
    pad = max(2, round(radius * 0.8))
    left = max(0, cx - radius - pad)
    right = min(gray.shape[1], cx + radius + pad + 1)
    top = max(0, cy - radius - pad)
    bottom = min(gray.shape[0], cy + radius + pad + 1)
    if right <= left or bottom <= top:
        return 0.0
    patch = gray[top:bottom, left:right].astype(np.float32)
    yy, xx = np.ogrid[top - cy : bottom - cy, left - cx : right - cx]
    d2 = xx * xx + yy * yy
    inner = d2 <= max(1.0, radius * 0.55) ** 2
    outer = (d2 >= (radius * 1.15) ** 2) & (d2 <= (radius * 1.85) ** 2)
    if not np.any(inner) or not np.any(outer):
        return 0.0
    inner_mean = float(patch[inner].mean())
    outer_mean = float(patch[outer].mean())
    return max(0.0, inner_mean - outer_mean if bright else outer_mean - inner_mean)


def _edge_ring_metrics(gradient: np.ndarray, cx: int, cy: int, radius: int) -> tuple[float, float]:
    radius = max(2, int(radius))
    pad = max(2, round(radius * 0.45))
    left = max(0, cx - radius - pad)
    right = min(gradient.shape[1], cx + radius + pad + 1)
    top = max(0, cy - radius - pad)
    bottom = min(gradient.shape[0], cy + radius + pad + 1)
    if right <= left or bottom <= top:
        return 0.0, 0.0
    patch = gradient[top:bottom, left:right].astype(np.float32)
    yy, xx = np.ogrid[top - cy : bottom - cy, left - cx : right - cx]
    gx = np.broadcast_to(xx.astype(np.float32), patch.shape)
    gy = np.broadcast_to(yy.astype(np.float32), patch.shape)
    distance = np.sqrt(gx * gx + gy * gy)
    ring = (distance >= radius * 0.70) & (distance <= radius * 1.35)
    if not np.any(ring):
        return 0.0, 0.0
    values = patch[ring]
    mean_strength = float(values.mean()) if values.size else 0.0
    threshold = max(18.0, mean_strength * 1.05)
    strong = ring & (patch >= threshold)
    coverage = _angular_coverage(strong, gx, gy, sectors=16)
    return mean_strength, coverage


def _angular_coverage(mask: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, *, sectors: int) -> float:
    if sectors <= 0 or not np.any(mask):
        return 0.0
    angles = np.arctan2(grid_y[mask], grid_x[mask])
    bins = np.floor(((angles + np.pi) / (2.0 * np.pi)) * sectors).astype(np.int32)
    bins = np.clip(bins, 0, sectors - 1)
    return float(np.unique(bins).size) / float(sectors)


def _refine_center(gray: np.ndarray, cx: int, cy: int, radius: int, *, bright: bool) -> tuple[float, float]:
    radius = max(2, int(radius))
    half = max(radius, round(radius * 1.15))
    left = max(0, cx - half)
    right = min(gray.shape[1], cx + half + 1)
    top = max(0, cy - half)
    bottom = min(gray.shape[0], cy + half + 1)
    if right <= left or bottom <= top:
        return float(cx), float(cy)
    patch = gray[top:bottom, left:right].astype(np.float32)
    lo = float(patch.min())
    hi = float(patch.max())
    if hi - lo < 4.0:
        return float(cx), float(cy)
    threshold = lo + (hi - lo) * 0.5
    weights = patch - threshold if bright else threshold - patch
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 1e-6:
        return float(cx), float(cy)
    ys = np.arange(patch.shape[0], dtype=np.float32).reshape(-1, 1)
    xs = np.arange(patch.shape[1], dtype=np.float32).reshape(1, -1)
    rx = float(left) + float((weights * xs).sum() / total)
    ry = float(top) + float((weights * ys).sum() / total)
    dist = float(np.hypot(rx - cx, ry - cy))
    max_move = radius * 0.8
    if dist > max_move and dist > 1e-6:
        scale = max_move / dist
        rx = float(cx) + (rx - float(cx)) * scale
        ry = float(cy) + (ry - float(cy)) * scale
    return rx, ry


def _intensity_gate(gray: np.ndarray, prepared: np.ndarray, scan: ViaPolarityScan) -> np.ndarray:
    if scan.low <= 0 and scan.high >= 255:
        return np.full(gray.shape[:2], 255, dtype=np.uint8)
    low = max(0, min(255, int(scan.low)))
    high = max(0, min(255, int(scan.high)))
    if low > high:
        low, high = high, low
    mask = np.where(((gray >= low) & (gray <= high)) | ((prepared >= low) & (prepared <= high)), 255, 0).astype(
        np.uint8
    )
    if cv2.countNonZero(mask) == 0:
        return np.full(gray.shape[:2], 255, dtype=np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.dilate(mask, k, iterations=1)


def _build_isolation(
    gate: np.ndarray,
    gray: np.ndarray,
    bright: bool,
    *,
    max_radius: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if gate.size == 0:
        return None
    mask_area = float(cv2.countNonZero(gate))
    if mask_area <= 0.0:
        return None
    if mask_area / float(gate.size) > 0.55:
        _threshold, otsu = cv2.threshold(to_gray_u8(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = otsu if bright else cv2.bitwise_not(otsu)
    else:
        mask = gate
    k = _odd(max_radius * 0.45, minimum=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(mask) <= 0:
        return None
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    return labels, stats


def _isolation_score(isolation: tuple[np.ndarray, np.ndarray] | None, cx: int, cy: int, radius: int) -> float:
    if isolation is None:
        return 1.0
    labels, stats = isolation
    if cy < 0 or cx < 0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
        return 0.0
    label = int(labels[cy, cx])
    if label <= 0 or label >= stats.shape[0]:
        return 0.35
    width = float(stats[label, cv2.CC_STAT_WIDTH])
    height = float(stats[label, cv2.CC_STAT_HEIGHT])
    diameter = float(radius * 2 + 1)
    elongation = max(width, height) / max(1.0, min(width, height))
    span = max(width, height)
    size_score = 1.0 if span <= diameter * 2.5 else max(0.0, 1.0 - (span - diameter * 2.5) / (diameter * 3.0))
    elongation_score = 1.0 if elongation <= 3.0 else max(0.0, 1.0 - (elongation - 3.0) / 5.0)
    return min(size_score, elongation_score)


def _extract_peaks(response: np.ndarray, *, min_distance: int) -> list[tuple[int, int]]:
    data = to_gray_u8(response)
    values = data[data > 0]
    if values.size == 0:
        return []
    cutoff = max(28, int(np.percentile(values, 92) * 0.65))
    radius = max(1, int(min_distance))
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    local_max = (data >= cv2.dilate(data, kernel)) & (data >= cutoff)
    ys, xs = np.where(local_max)
    if len(xs) == 0:
        return []
    order = np.argsort(data[ys, xs])[::-1]
    used = np.zeros(data.shape, dtype=bool)
    peaks: list[tuple[int, int]] = []
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if used[y, x]:
            continue
        peaks.append((y, x))
        y0 = max(0, y - radius)
        y1 = min(data.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(data.shape[1], x + radius + 1)
        used[y0:y1, x0:x1] = True
    return peaks


def _nms(candidates: list[_ViaCandidate], *, iou_threshold: float) -> list[_ViaCandidate]:
    kept: list[_ViaCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(_candidate_iou(candidate, existing) >= iou_threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _candidate_iou(first: _ViaCandidate, second: _ViaCandidate) -> float:
    ax0 = first.center_x - first.width * 0.5
    ay0 = first.center_y - first.height * 0.5
    ax1 = ax0 + first.width
    ay1 = ay0 + first.height
    bx0 = second.center_x - second.width * 0.5
    by0 = second.center_y - second.height * 0.5
    bx1 = bx0 + second.width
    by1 = by0 + second.height
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = first.width * first.height + second.width * second.height - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _candidate_radii(config: SemPrimaryViaConfig) -> list[int]:
    diameters = [float(value) for value in config.expected_diameters if int(value) > 0]
    if not diameters:
        lo = max(3.0, min(config.min_width, config.min_height))
        hi = max(lo, min(config.max_width, config.max_height, max(12.0, lo * 3.0)))
        step = max(1.0, (hi - lo) / 4.0)
        value = lo
        while value <= hi + 0.5:
            diameters.append(value)
            value += step
    return sorted({max(2, round(diameter / 2.0)) for diameter in diameters})


def _expected_diameters(settings: Any) -> list[int]:
    values: list[int] = []
    for width, height in zip(
        getattr(settings, "fixed_via_widths", []) or [],
        getattr(settings, "fixed_via_heights", []) or [],
        strict=False,
    ):
        if int(width) > 0 and int(height) > 0:
            values.append(max(3, round((int(width) + int(height)) / 2.0)))
    for attr in ("min_via_width", "min_via_height", "max_via_width", "max_via_height"):
        value = getattr(settings, attr, None)
        if value not in (None, "", 0, 0.0):
            values.append(max(3, int(value)))
    if not values:
        values.append(9)
    median = max(3, round(float(np.median(np.asarray(values, dtype=np.float32)))))
    return sorted({median, *values})[:6]


def _size_limits(settings: Any, expected: list[int]) -> tuple[float, float, float, float]:
    mode = str(getattr(settings, "via_size_mode", "range") or "range").strip().lower()
    widths = [float(value) for value in getattr(settings, "fixed_via_widths", []) or [] if int(value) > 0]
    heights = [float(value) for value in getattr(settings, "fixed_via_heights", []) or [] if int(value) > 0]
    if mode == "fixed" and widths and heights:
        return min(widths) * 0.45, max(widths) * 1.65, min(heights) * 0.45, max(heights) * 1.65
    exp = float(max(3, round(float(np.median(np.asarray(expected, dtype=np.float32)))))) if expected else 9.0
    min_w = float(getattr(settings, "min_via_width", 0) or max(2.0, exp * 0.35))
    min_h = float(getattr(settings, "min_via_height", 0) or max(2.0, exp * 0.35))
    max_w = float(getattr(settings, "max_via_width", None) or max(12.0, exp * 3.5))
    max_h = float(getattr(settings, "max_via_height", None) or max(12.0, exp * 3.5))
    return min_w, max_w, min_h, max_h


def _polarity_scans(settings: Any) -> list[ViaPolarityScan]:
    scans: list[ViaPolarityScan] = []
    if bool(getattr(settings, "via_white_range_enabled", True)):
        scans.append(
            ViaPolarityScan(
                bright=True,
                low=max(0, min(255, int(getattr(settings, "via_white_range_min", 180)))),
                high=max(0, min(255, int(getattr(settings, "via_white_range_max", 255)))),
            )
        )
    if bool(getattr(settings, "via_black_range_enabled", False)):
        scans.append(
            ViaPolarityScan(
                bright=False,
                low=max(0, min(255, int(getattr(settings, "via_black_range_min", 0)))),
                high=max(0, min(255, int(getattr(settings, "via_black_range_max", 60)))),
            )
        )
    if not scans:
        scans = [ViaPolarityScan(bright=True), ViaPolarityScan(bright=False)]
    return scans


def _normalize_float(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.zeros_like(data, dtype=np.uint8)
    data = np.nan_to_num(data.astype(np.float32), copy=False)
    max_value = float(np.max(np.abs(data)))
    if max_value <= 1e-6:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip(data * (255.0 / max_value), 0.0, 255.0).astype(np.uint8)


def _odd(value: float, *, minimum: int) -> int:
    result = max(int(minimum), round(float(value)))
    if result % 2 == 0:
        result += 1
    return result
