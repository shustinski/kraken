"""Multi-stage metal segmentation for conductor recovery (SEM-aware)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ...utils import ensure_binary_mask, ensure_uint8
from ..preprocessing import NoiseLevel, PreprocessConfig, guess_polarity, preprocess_for_sem
from ..schemas import SemPolarity


class MetalSegmentationStrategy(StrEnum):
    AUTO = "auto"
    LOCAL_ADAPTIVE = "local_adaptive"
    SAUVOLA = "sauvola"
    EDGES = "edges"
    LEGACY_OTSU = "legacy_otsu"


def normalize_metal_segmentation_strategy(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "auto": MetalSegmentationStrategy.AUTO,
        "local_adaptive": MetalSegmentationStrategy.LOCAL_ADAPTIVE,
        "local": MetalSegmentationStrategy.LOCAL_ADAPTIVE,
        "adaptive": MetalSegmentationStrategy.LOCAL_ADAPTIVE,
        "адаптивная": MetalSegmentationStrategy.LOCAL_ADAPTIVE,
        "sauvola": MetalSegmentationStrategy.SAUVOLA,
        "sauvola_like": MetalSegmentationStrategy.SAUVOLA,
        "edges": MetalSegmentationStrategy.EDGES,
        "edge": MetalSegmentationStrategy.EDGES,
        "none": MetalSegmentationStrategy.EDGES,
        "без": MetalSegmentationStrategy.EDGES,
        "без_сегментации": MetalSegmentationStrategy.EDGES,
        "без сегментации": MetalSegmentationStrategy.EDGES,
        "grayscale": MetalSegmentationStrategy.EDGES,
        "legacy_otsu": MetalSegmentationStrategy.LEGACY_OTSU,
        "otsu": MetalSegmentationStrategy.LEGACY_OTSU,
        "hybrid": MetalSegmentationStrategy.AUTO,
        "гибрид": MetalSegmentationStrategy.AUTO,
        "гибридная": MetalSegmentationStrategy.AUTO,
    }
    return str(mapping.get(text, MetalSegmentationStrategy.AUTO))


def contrast_bias_to_threshold_params(contrast_bias: float) -> dict[str, float]:
    """Map UI contrast bias (-50..+50) to continuous threshold knobs."""
    bias = max(-50.0, min(50.0, float(contrast_bias)))
    t = (bias + 50.0) / 100.0
    c_adaptive = 2.0 + (1.0 - t) * 14.0
    sauvola_k = 0.06 + (1.0 - t) * 0.14
    otsu_offset = (t - 0.5) * 18.0
    return {
        "c_adaptive": float(c_adaptive),
        "sauvola_k": float(sauvola_k),
        "otsu_offset": float(otsu_offset),
    }


def noise_suppression_to_preprocess(noise_suppression: int) -> PreprocessConfig:
    n = max(0, min(100, int(noise_suppression)))
    if n < 25:
        level = NoiseLevel.LOW
        clahe_clip = 1.6
    elif n < 55:
        level = NoiseLevel.MEDIUM
        clahe_clip = 2.0
    else:
        level = NoiseLevel.HIGH
        clahe_clip = 2.4 + (n - 55) / 45.0 * 1.2
    bg_frac = 0.03 + (n / 100.0) * 0.03
    return PreprocessConfig(
        clahe_clip=float(clahe_clip),
        clahe_grid=8,
        subtract_background=True,
        background_sigma_fraction=float(bg_frac),
        denoise=level,
    )


def migrate_legacy_metal_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert deprecated sensitivity/segmentation fields to new pipeline params."""
    out = dict(payload)
    if "metal_contrast_bias" not in payload and "metal_sensitivity_0_100" in payload:
        sens = max(0, min(100, int(payload.get("metal_sensitivity_0_100", 50))))
        tok = str(payload.get("metal_sensitivity", "medium") or "medium").lower()
        mid = {"low": 35, "medium": 50, "high": 65}.get(tok, 50)
        blend = 0.35 * mid + 0.65 * sens
        out["metal_contrast_bias"] = int(round((blend - 50.0) * 0.6))
    if "metal_segmentation_strategy" not in payload:
        old = str(payload.get("metal_segmentation_method", "otsu") or "otsu").lower()
        if old in {"none", "edges", "edge", "grayscale"}:
            out["metal_segmentation_strategy"] = "edges"
        elif old == "otsu":
            out["metal_segmentation_strategy"] = "auto"
        elif old == "adaptive":
            out["metal_segmentation_strategy"] = "local_adaptive"
        elif old == "hybrid":
            out["metal_segmentation_strategy"] = "auto"
        else:
            out["metal_segmentation_strategy"] = "auto"
    if "metal_gap_bridge_px" not in payload and "metal_morph_close_radius" in payload:
        out["metal_gap_bridge_px"] = int(payload.get("metal_morph_close_radius", 2) or 2)
    if "metal_speckle_removal_px" not in payload and "metal_morph_open_radius" in payload:
        out["metal_speckle_removal_px"] = int(payload.get("metal_morph_open_radius", 0) or 0)
    if "metal_noise_suppression" not in payload:
        preset = str(payload.get("metal_preset", "standard") or "standard")
        preset_noise = {"standard": 20, "noisy_sem": 70, "thin_traces": 40, "wide_traces": 30}.get(preset, 30)
        out["metal_noise_suppression"] = preset_noise
    return out


@dataclass(slots=True)
class MetalSegmentationConfig:
    noise_suppression: int = 20
    contrast_bias: float = 0.0
    segmentation_strategy: str = "auto"
    gap_bridge_px: int = 2
    speckle_removal_px: int = 0
    min_width_px: float = 8.0
    edge_close_cap_px: int = 9
    edge_watershed_split: bool = True
    edge_watershed_dist_peak_frac: float = 0.38
    edge_watershed_max_pixels: int | None = 3_000_000
    block_size: int = 35
    sauvola_window: int = 31
    min_component_area: int = 20
    max_hole_fill_area: int = 200


@dataclass(slots=True)
class MetalSegmentationResult:
    mask: np.ndarray
    preprocessed: np.ndarray
    raw_segmentation: np.ndarray
    after_topology: np.ndarray
    strategy: str
    polarity: SemPolarity
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


def _odd(n: int) -> int:
    m = int(n)
    if m % 2 == 0:
        m += 1
    return max(3, m)


def _sauvola_bin(gray: np.ndarray, window: int, k: float, dark_foreground: bool) -> np.ndarray:
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
        return (g < t).astype(np.uint8) * 255
    return (g > t).astype(np.uint8) * 255


def _adaptive(gray: np.ndarray, method: int, block: int, c: float, invert: bool) -> np.ndarray:
    b = _odd(block)
    raw = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, b, int(round(c)))
    if invert:
        raw = cv2.bitwise_not(raw)
    return raw


def _legacy_otsu_mask(gray: np.ndarray, otsu_offset: float) -> np.ndarray:
    otsu_t, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if abs(otsu_offset) > 0.05:
        t = int(max(1, min(254, round(float(otsu_t) + otsu_offset))))
        _, mask = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return mask


def _canny_closed_fill(gray: np.ndarray, dark_fg: bool, contrast_bias: float) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    adj = max(-50.0, min(50.0, float(contrast_bias)))
    scale = max(0.55, 1.0 - adj / 80.0)
    lo = int(max(8, min(40, round(18 * scale))))
    hi = int(max(lo + 8, min(90, round(55 * scale))))
    edges = cv2.Canny(blur, lo, hi)
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


def _fill_small_holes(mask: np.ndarray, *, max_area: int) -> np.ndarray:
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


def _count_binary_components(mask_u8: np.ndarray) -> int:
    if mask_u8 is None or mask_u8.size == 0:
        return 0
    m = (np.asarray(mask_u8) > 0).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return max(0, int(n) - 1)


def _watershed_split_touching_conductors(
    ribbon_u8: np.ndarray,
    guide_bgr: np.ndarray,
    *,
    dist_peak_frac: float,
) -> np.ndarray:
    m = (np.asarray(ribbon_u8) > 0).astype(np.uint8) * 255
    if int(cv2.countNonZero(m)) < 80:
        return ribbon_u8
    before_cc = _count_binary_components(m)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dmax = float(np.max(dist))
    if dmax < 2.2:
        return ribbon_u8
    frac = max(0.22, min(0.55, float(dist_peak_frac)))
    _, sure_fg = cv2.threshold(dist, frac * dmax, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    if int(cv2.countNonZero(sure_fg)) < 16:
        return ribbon_u8
    unknown = cv2.subtract(m, sure_fg)
    n_mark, markers = cv2.connectedComponents(sure_fg)
    if n_mark < 3:
        return ribbon_u8
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0
    markers[m == 0] = 0
    img = np.asarray(guide_bgr)
    if img.ndim == 2:
        img3 = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img3 = img[:, :, :3].astype(np.uint8)
    else:
        img3 = img.astype(np.uint8)
    ws = markers.copy()
    raise_if_preview_cancelled()
    cv2.watershed(img3, ws)
    raise_if_preview_cancelled()
    out = np.zeros_like(m, dtype=np.uint8)
    for lbl in np.unique(ws):
        li = int(lbl)
        if li <= 1 or li == -1:
            continue
        out[ws == li] = 255
    if int(cv2.countNonZero(out)) < int(0.22 * float(cv2.countNonZero(m))):
        return ribbon_u8
    after_cc = _count_binary_components(out)
    if after_cc <= before_cc:
        return ribbon_u8
    if after_cc > max(before_cc + 8, before_cc * 3):
        return ribbon_u8
    return ensure_binary_mask(out)


def _filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = (ensure_uint8(mask) > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == i] = 255
    return out


def _grayscale_edge_mask(
    gray: np.ndarray,
    config: MetalSegmentationConfig,
    *,
    contrast_bias: float,
) -> tuple[np.ndarray, np.ndarray]:
    if gray.size == 0:
        z = np.zeros_like(gray)
        return z, z
    gh, gw = int(gray.shape[0]), int(gray.shape[1])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ksz = max(5, min(21, int(2.25 * max(2.0, float(config.min_width_px))) | 1))
    k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    tophat = cv2.subtract(blurred, cv2.morphologyEx(blurred, cv2.MORPH_OPEN, k_th))
    enhanced = cv2.addWeighted(blurred, 0.58, tophat, 0.42, 0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    med = float(np.median(enhanced))
    if med < 1.0:
        med = float(np.mean(enhanced)) + 1.0
    sigma_use = 0.33
    lower = float((1.0 - sigma_use) * med)
    upper = float((1.0 + sigma_use) * med)
    adj = float(contrast_bias) / 50.0
    lower *= max(0.35, 1.0 - 0.28 * adj)
    upper *= max(0.55, 1.0 - 0.18 * adj)
    lo = int(max(1, min(254, round(lower))))
    hi = int(max(lo + 4, min(255, round(upper))))
    edges = cv2.Canny(enhanced, lo, hi, L2gradient=True)
    raise_if_preview_cancelled()

    d3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick = cv2.dilate(edges, d3, iterations=1)
    close_r = max(1, int(config.gap_bridge_px))
    rk = max(3, min(25, close_r * 2 + 1))
    inner_merge = int(max(5, min(15, 2 * int(max(2, round(0.42 * float(config.min_width_px)))) + 1)))
    cap = max(5, min(21, int(config.edge_close_cap_px) | 1))
    rw = min(rk, inner_merge, cap)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rw, rw))
    ribbon = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, k_close)
    raise_if_preview_cancelled()
    open_r = max(0, int(config.speckle_removal_px))
    if open_r > 0:
        ko = max(3, open_r * 2 + 1)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
        ribbon = cv2.morphologyEx(ribbon, cv2.MORPH_OPEN, k_open)
    if config.edge_watershed_split:
        cap_px = config.edge_watershed_max_pixels
        run_ws = cap_px is None or int(cap_px) <= 0 or gh * gw <= int(cap_px)
        if run_ws:
            guide = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            ribbon = _watershed_split_touching_conductors(
                ribbon,
                guide,
                dist_peak_frac=float(config.edge_watershed_dist_peak_frac),
            )
    return ensure_binary_mask(ribbon), edges


def _raw_segmentation(
    prep: np.ndarray,
    *,
    strategy: str,
    thresh: dict[str, float],
    dark_fg: bool,
    invert_a: bool,
    config: MetalSegmentationConfig,
) -> tuple[np.ndarray, str, dict[str, np.ndarray]]:
    block = _odd(config.block_size)
    c_ad = float(thresh["c_adaptive"])
    sk = float(thresh["sauvola_k"])
    extra: dict[str, np.ndarray] = {}
    strat = normalize_metal_segmentation_strategy(strategy)

    raw_adaptive = _adaptive(prep, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block, c_ad, invert_a)
    raw_sauvola = _sauvola_bin(prep, config.sauvola_window, sk, dark_fg)
    raw_otsu = _legacy_otsu_mask(prep, float(thresh["otsu_offset"]))
    if not dark_fg:
        raw_otsu = cv2.bitwise_not(raw_otsu)
    raw_edges, canny = _grayscale_edge_mask(
        prep,
        config,
        contrast_bias=float(config.contrast_bias),
    )
    extra["metal_edge_canny"] = canny

    if strat == MetalSegmentationStrategy.LOCAL_ADAPTIVE:
        return raw_adaptive, "local_adaptive", extra
    if strat == MetalSegmentationStrategy.SAUVOLA:
        return raw_sauvola, "sauvola", extra
    if strat == MetalSegmentationStrategy.LEGACY_OTSU:
        return raw_otsu, "legacy_otsu", extra
    if strat == MetalSegmentationStrategy.EDGES:
        return raw_edges, "edges", extra

    candidates: list[tuple[str, float, np.ndarray]] = []
    for name, raw in (
        ("local_adaptive", raw_adaptive),
        ("sauvola", raw_sauvola),
        ("legacy_otsu", raw_otsu),
        ("edges", raw_edges),
    ):
        q = _quality_score(prep, raw)
        candidates.append((name, q, raw))
    candidates.sort(key=lambda t: t[1], reverse=True)
    best_name, _, best = candidates[0]
    return best, best_name, extra


def apply_topology_repair(raw_mask: np.ndarray, config: MetalSegmentationConfig) -> np.ndarray:
    m = ensure_uint8(raw_mask)
    close_r = max(0, int(config.gap_bridge_px))
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = _fill_small_holes(m, max_area=int(config.max_hole_fill_area))
    open_r = max(0, int(config.speckle_removal_px))
    if open_r > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    return ensure_binary_mask(m)


def build_metal_segmentation_mask(gray: np.ndarray, config: MetalSegmentationConfig) -> MetalSegmentationResult:
    g0 = ensure_uint8(gray)
    if g0.size == 0:
        z = np.zeros_like(g0)
        return MetalSegmentationResult(
            mask=z,
            preprocessed=z,
            raw_segmentation=z,
            after_topology=z,
            strategy="empty",
            polarity=SemPolarity.AUTO,
        )

    prep_cfg = noise_suppression_to_preprocess(config.noise_suppression)
    prep = preprocess_for_sem(g0, prep_cfg)
    raise_if_preview_cancelled()

    pol = guess_polarity(prep)
    if pol == SemPolarity.AUTO:
        pol = guess_polarity(prep)
    dark_fg = pol is SemPolarity.DARK_FOREGROUND
    invert_a = not dark_fg
    thresh = contrast_bias_to_threshold_params(config.contrast_bias)

    raw, strategy, extra = _raw_segmentation(
        prep,
        strategy=config.segmentation_strategy,
        thresh=thresh,
        dark_fg=dark_fg,
        invert_a=invert_a,
        config=config,
    )
    raise_if_preview_cancelled()

    after_topo = apply_topology_repair(raw, config)
    mask = _filter_components(after_topo, int(config.min_component_area))

    debug = {
        "metal_preprocessed": prep,
        "metal_raw_segmentation": ensure_binary_mask(raw),
        "metal_after_topology": ensure_binary_mask(after_topo),
        **extra,
    }
    return MetalSegmentationResult(
        mask=mask,
        preprocessed=prep,
        raw_segmentation=ensure_binary_mask(raw),
        after_topology=ensure_binary_mask(after_topo),
        strategy=strategy,
        polarity=pol,
        debug_images=debug,
    )
