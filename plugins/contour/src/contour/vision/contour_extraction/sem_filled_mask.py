"""
Filled-region segmentation for SEM (not edge-only): outputs a *paint-style* uint8 mask.

Design goals:
- avoid single global Otsu as the only path;
- use adaptive *local* thresholds on illumination-flattened data;
- auto-select or blend strategies using a cheap quality score;
- use conservative morphology (small close for connectivity; avoid big openings that
  swallow holes — interior voids are preserved via ``RETR_TREE`` on the mask).

*Assumption:* foreground objects are either darker or brighter than the background
in a *locally* consistent way after preprocessing (``guess_polarity`` in preprocessing).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from ..preprocessing import PreprocessConfig, guess_polarity, preprocess_for_sem
from ..schemas import SemPolarity

try:
    from ...utils import ensure_uint8
except ImportError:  # pragma: no cover
    from ..io_normalize import ensure_uint8_local as ensure_uint8  # type: ignore[misc,assignment]


class SegmentationStrategyName(StrEnum):
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSS = "adaptive_gauss"
    SAUVOLA = "sauvola_like"
    CEDGE_FILL = "cedge_fill"


@dataclass(frozen=True, slots=True)
class FilledMaskSegmentationConfig:
    """Internal tuning; user selects only :class:`SegPreset` in UI."""

    block_size: int = 35
    c_adaptive: int = 7
    sauvola_window: int = 31
    sauvola_k: float = 0.12
    close_radius: int = 2
    min_component_area: int = 20
    max_hole_fill_area: int = 200  # only tiny speckles, not true interior holes


@dataclass(frozen=True, slots=True)
class SegPreset:
    id: str
    noise: str  # "low" | "medium" | "high" — maps to :class:`preprocessing.NoiseLevel`


@dataclass(slots=True)
class FilledMaskResult:
    mask: np.ndarray
    strategy: str
    alternatives: list[tuple[str, float]]
    preprocessed: np.ndarray
    polarity: SemPolarity


def _odd(n: int) -> int:
    m = int(n)
    if m % 2 == 0:
        m += 1
    return max(3, m)


def _sauvola_bin(gray: np.ndarray, window: int, k: float, dark_foreground: bool) -> np.ndarray:
    """Local Sauvola (OpenCV box blur); R fixed at half dynamic range in float [0,1]."""

    g = gray.astype(np.float32) / 255.0
    w = _odd(window)
    ksize = (w, w)
    m = cv2.blur(g, ksize, borderType=cv2.BORDER_REPLICATE)
    m2 = cv2.blur(g * g, ksize, borderType=cv2.BORDER_REPLICATE)
    v = np.clip(m2 - m * m, 0.0, None)
    s = np.sqrt(v.astype(np.float32))
    r = 0.5
    t = m * (1.0 + float(k) * (s / (r + 1e-6) - 1.0))
    if dark_foreground:
        b = (g < t).astype(np.uint8) * 255
    else:
        b = (g > t).astype(np.uint8) * 255
    return b


def _adaptive(gray: np.ndarray, method: int, block: int, c: int, invert: bool) -> np.ndarray:
    b = _odd(block)
    raw = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, b, c)
    if invert:
        raw = cv2.bitwise_not(raw)
    return raw


def _canny_closed_fill(gray: np.ndarray, dark_fg: bool) -> np.ndarray:
    """Fallback: strong-gradient ridge → closed boundary → fill *outside*; interior becomes FG."""

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    edges = cv2.Canny(blur, 18, 55)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, iterations=2)
    h, s = edges.shape
    inv = 255 - edges
    border = np.zeros((h + 2, s + 2), dtype=np.uint8)
    inv_copy = inv.copy()
    cv2.floodFill(inv_copy, border, (0, 0), 0)
    outside = (inv == inv_copy).astype(np.uint8) * 255
    fg = cv2.bitwise_not(outside)
    if not dark_fg:
        fg = cv2.bitwise_not(fg)
    return fg


def _morphology_conservative(mask: np.ndarray, close_radius: int, max_hole_area: int) -> np.ndarray:
    m = ensure_uint8(mask)
    r = max(0, int(close_radius))
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = _fill_small_holes(m, max_area=max_hole_area)
    return m


def _fill_small_holes(mask: np.ndarray, *, max_area: int) -> np.ndarray:
    """Fill only *small* interior holes; keeps larger nested regions."""

    m = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(m) == 0 or cv2.countNonZero(m) == m.size:
        return m
    inv = cv2.bitwise_not(m)
    h, s = m.shape
    border = np.zeros((h + 2, s + 2), dtype=np.uint8)
    inv_copy = inv.copy()
    cv2.floodFill(inv_copy, border, (0, 0), 255)
    holes = cv2.subtract(inv, inv_copy)
    if cv2.countNonZero(holes) == 0:
        return m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) <= max_area:
            holes[labels == i] = 0
    return cv2.subtract(m, holes)


def _quality_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """Heuristic: boundary gradient energy + local contrast inside mask (higher = sharper)."""

    m = (ensure_uint8(mask) > 0).astype(np.float32)
    if float(m.sum()) < 1.0:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    b = m - cv2.blur(m, (5, 5))
    br = (np.abs(b) > 0.15).astype(np.float32)
    if float(br.sum()) < 1.0:
        edge = float((mag * m).sum() / (float(m.sum()) + 1e-6))
    else:
        edge = float((mag * br).sum() / (float(br.sum()) + 1e-6))
    g = gray.astype(np.float32)
    mu = float((g * m).sum() / (float(m.sum()) + 1e-6))
    out_mu = float((g * (1.0 - m)).sum() / (float((1.0 - m).sum()) + 1e-6))
    contrast = abs(mu - out_mu) / 255.0
    return edge * 0.55 + contrast * 0.45


def _filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = (ensure_uint8(mask) > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == i] = 255
    return out


def extract_filled_mask(
    gray: np.ndarray,
    *,
    config: FilledMaskSegmentationConfig,
    preprocess: PreprocessConfig,
    polarity: SemPolarity | None = None,
) -> FilledMaskResult:
    g0 = ensure_uint8(gray)
    prep = preprocess_for_sem(g0, preprocess)
    pol = polarity or guess_polarity(prep)
    if pol == SemPolarity.AUTO:
        pol = guess_polarity(prep)
    dark_fg = pol is SemPolarity.DARK_FOREGROUND
    invert_a = not dark_fg

    strategies: list[tuple[str, np.ndarray]] = []
    block = _odd(config.block_size)
    # Strategies differ only in the binarization core; shared conservative morphology after scoring.
    raw_mean = _adaptive(prep, cv2.ADAPTIVE_THRESH_MEAN_C, block, int(config.c_adaptive), invert_a)
    raw_gss = _adaptive(prep, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block, int(config.c_adaptive), invert_a)
    raw_s = _sauvola_bin(prep, window=int(config.sauvola_window), k=float(config.sauvola_k), dark_foreground=dark_fg)
    raw_c = _canny_closed_fill(prep, dark_fg)

    for name, raw in [
        (SegmentationStrategyName.ADAPTIVE_MEAN, raw_mean),
        (SegmentationStrategyName.ADAPTIVE_GAUSS, raw_gss),
        (SegmentationStrategyName.SAUVOLA, raw_s),
        (SegmentationStrategyName.CEDGE_FILL, raw_c),
    ]:
        mm = _morphology_conservative(raw, config.close_radius, config.max_hole_fill_area)
        mm = _filter_components(mm, int(config.min_component_area))
        strategies.append((name, mm))

    scored: list[tuple[str, float, np.ndarray]] = []
    for name, m in strategies:
        q = _quality_score(prep, m)
        scored.append((name, q, m))
    scored.sort(key=lambda t: t[1], reverse=True)
    best_name, _best_q, best_mask = scored[0]
    alternatives = [(n, float(q)) for n, q, _ in scored]

    return FilledMaskResult(
        mask=best_mask,
        strategy=best_name,
        alternatives=alternatives,
        preprocessed=prep,
        polarity=pol,
    )


def label_segmentation_strategies() -> list[str]:
    return [m.value for m in SegmentationStrategyName]
