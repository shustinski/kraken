"""Marker watershed segmentation for SEM conductors outlined by bright rims.

Global thresholding fails on these images because the bright rims form a third
intensity class: Otsu then splits rim from fill instead of metal from substrate.
Here the two reliable populations seed a watershed instead, and the region
boundary is placed on the intensity edge between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ...utils import ensure_binary_mask, ensure_uint8


@dataclass(frozen=True, slots=True)
class GradientWatershedConfig:
    """Seed levels are expressed as margins around the two Otsu class limits."""

    smoothing_sigma: float = 1.0
    core_margin: float = 8.0
    groove_margin: float = 16.0
    rim_probe_px: int = 6
    seed_speckle_px: int = 1
    valley_span_px: int = 5
    valley_depth: float = 45.0
    random_walker_beta: float = 90.0
    random_walker_iterations: int = 160
    graph_cut_iterations: int = 5
    reconstruction_erode_px: int = 0


def clamped_gradient_watershed_config(
    *,
    smoothing_sigma: float = 1.0,
    core_margin: float = 8.0,
    groove_margin: float = 16.0,
    rim_probe_px: int = 6,
    seed_speckle_px: int = 1,
    valley_span_px: int = 5,
    valley_depth: float = 45.0,
    random_walker_beta: float = 90.0,
    random_walker_iterations: int = 160,
    graph_cut_iterations: int = 5,
    reconstruction_erode_px: int = 0,
) -> GradientWatershedConfig:
    return GradientWatershedConfig(
        smoothing_sigma=max(0.1, min(8.0, float(smoothing_sigma))),
        core_margin=max(0.0, min(40.0, float(core_margin))),
        groove_margin=max(0.0, min(40.0, float(groove_margin))),
        rim_probe_px=max(1, min(32, int(rim_probe_px))),
        seed_speckle_px=max(0, min(8, int(seed_speckle_px))),
        valley_span_px=max(0, min(16, int(valley_span_px))),
        valley_depth=max(0.0, min(160.0, float(valley_depth))),
        random_walker_beta=max(1.0, min(400.0, float(random_walker_beta))),
        random_walker_iterations=max(8, min(400, int(random_walker_iterations))),
        graph_cut_iterations=max(1, min(16, int(graph_cut_iterations))),
        reconstruction_erode_px=max(0, min(16, int(reconstruction_erode_px))),
    )


def gradient_watershed_config_from_object(source: object) -> GradientWatershedConfig:
    return clamped_gradient_watershed_config(
        smoothing_sigma=float(getattr(source, "watershed_smoothing_sigma", 1.0) or 1.0),
        core_margin=float(getattr(source, "watershed_core_margin", 8.0) or 0.0),
        groove_margin=float(getattr(source, "watershed_groove_margin", 16.0) or 0.0),
        rim_probe_px=int(getattr(source, "watershed_rim_probe_px", 6) or 1),
        seed_speckle_px=int(getattr(source, "watershed_seed_speckle_px", 1) or 0),
        valley_span_px=int(getattr(source, "watershed_valley_span_px", 5) or 0),
        valley_depth=float(getattr(source, "watershed_valley_depth", 45.0) or 0.0),
        random_walker_beta=float(getattr(source, "random_walker_beta", 90.0) or 90.0),
        random_walker_iterations=int(getattr(source, "random_walker_iterations", 160) or 160),
        graph_cut_iterations=int(getattr(source, "graph_cut_iterations", 5) or 5),
        reconstruction_erode_px=int(getattr(source, "reconstruction_erode_px", 0) or 0),
    )


def _otsu_level(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    level, _mask = cv2.threshold(
        values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(level)


def intensity_class_limits(smoothed: np.ndarray) -> tuple[float, float]:
    """Return (substrate limit, metal limit) from a two-level Otsu split."""
    metal_limit = _otsu_level(smoothed)
    substrate_limit = _otsu_level(smoothed[smoothed <= metal_limit])
    return substrate_limit, metal_limit


def _open_seeds(seeds: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return seeds
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(seeds, cv2.MORPH_OPEN, kernel)


def keep_rim_lined_seeds(
    seeds: np.ndarray,
    smoothed: np.ndarray,
    *,
    rim_level: float,
    probe_px: int,
) -> np.ndarray:
    """Drop dark seeds that sit inside metal instead of in a gap between conductors.

    A real gap is lined with the bright rims of its two neighbours, so the ring
    around the seed is bright.  Dark surface texture inside a wide pour is
    surrounded by fill and would otherwise shred that pour into fragments.
    """
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    if count <= 1:
        return seeds
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * probe_px + 1, 2 * probe_px + 1))
    grown = cv2.dilate(labels.astype(np.float32), kernel).astype(np.int32)
    ring = (grown > 0) & (labels == 0)
    if not np.any(ring):
        return seeds
    ring_labels = grown[ring]
    ring_values = smoothed[ring].astype(np.float64)
    sums = np.bincount(ring_labels, weights=ring_values, minlength=count)
    counts = np.bincount(ring_labels, minlength=count).astype(np.float64)
    ring_means = np.divide(sums, np.maximum(counts, 1.0))
    keep = ring_means >= rim_level
    keep |= stats[:, cv2.CC_STAT_AREA] >= 400
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    keep[border_labels] = True
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def keep_sandwiched_valley_seeds(
    seams: np.ndarray,
    smoothed: np.ndarray,
    *,
    span_px: int,
    flank_delta: float,
) -> np.ndarray:
    """Keep valleys that have brighter metal on two opposite sides.

    A real seam sits between conductors.  The body of a uniform trace is not
    sandwiched by brighter metal: its flanks are the darker substrate, so it
    must not become a gap seed or the whole polygon disappears.
    """
    if not np.any(seams) or span_px <= 0 or flank_delta <= 0.0:
        return np.zeros(seams.shape, dtype=np.uint8)
    span = max(1, int(span_px))
    center = smoothed.astype(np.int16)
    left = np.roll(center, span, axis=1)
    right = np.roll(center, -span, axis=1)
    up = np.roll(center, span, axis=0)
    down = np.roll(center, -span, axis=0)
    delta = np.int16(max(1.0, float(flank_delta)))
    horizontal = (left >= center + delta) & (right >= center + delta)
    vertical = (up >= center + delta) & (down >= center + delta)
    horizontal[:, :span] = False
    horizontal[:, -span:] = False
    vertical[:span, :] = False
    vertical[-span:, :] = False
    return np.where((seams > 0) & (horizontal | vertical), 255, 0).astype(np.uint8)


def keep_thin_valley_components(seams: np.ndarray, max_radius: float) -> np.ndarray:
    """Drop valley blobs thicker than a gap, such as the interior of a trace."""
    if not np.any(seams) or max_radius <= 0.0:
        return np.zeros(seams.shape, dtype=np.uint8)
    binary = (seams > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return seams
    thickest = np.zeros(count, dtype=np.float64)
    np.maximum.at(thickest, labels.ravel(), dist.ravel())
    keep = thickest <= float(max_radius)
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _cores_covering_fill(
    smoothed: np.ndarray,
    *,
    strong_cores: np.ndarray,
    groove_seeds: np.ndarray,
    weak_level: float,
    speckle: int,
) -> np.ndarray:
    """Seed bright rims and the mid-grey fill that is not already a gap."""
    fill = ((smoothed >= weak_level) & (groove_seeds == 0)).astype(np.uint8) * 255
    cores = cv2.bitwise_or(fill, strong_cores)
    return _open_seeds(cv2.subtract(cores, groove_seeds), speckle)


def narrow_valley_seeds(
    smoothed: np.ndarray,
    *,
    span_px: int,
    depth: float,
) -> np.ndarray:
    """Seed the thin dark seams that separate closely spaced conductors.

    Such a seam never reaches the substrate level, so an absolute dark threshold
    misses it and the two neighbours fuse.  Closing the image with a kernel wider
    than the seam lifts its floor to the flanking metal, and the difference marks
    the seam while leaving wide gaps and shallow surface texture untouched.
    """
    if span_px <= 0 or depth <= 0.0:
        return np.zeros(smoothed.shape, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * span_px + 1, 2 * span_px + 1))
    closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
    valley = closed.astype(np.int16) - smoothed.astype(np.int16)
    return (valley >= depth).astype(np.uint8) * 255


@dataclass(frozen=True, slots=True)
class ConductorSeeds:
    """Metal-core and gap markers shared by seeded conductor algorithms."""

    smoothed: np.ndarray
    core_seeds: np.ndarray
    groove_seeds: np.ndarray
    metal_limit: float

    @property
    def has_both_classes(self) -> bool:
        return bool(np.any(self.core_seeds)) and bool(np.any(self.groove_seeds))

    def fallback_mask(self) -> np.ndarray:
        return ensure_binary_mask((self.smoothed >= self.metal_limit).astype(np.uint8) * 255)


def build_conductor_seeds(gray: np.ndarray, config: GradientWatershedConfig) -> ConductorSeeds | None:
    """Build rim-lit metal cores and gap seeds, or None for an empty frame."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return None

    smoothed = cv2.GaussianBlur(source, (0, 0), max(0.1, float(config.smoothing_sigma)))
    substrate_limit, metal_limit = intensity_class_limits(smoothed)
    core_level = metal_limit - float(config.core_margin)
    groove_level = min(substrate_limit + float(config.groove_margin), metal_limit - 4.0)

    speckle = max(0, int(config.seed_speckle_px))
    strong_cores = _open_seeds((smoothed >= core_level).astype(np.uint8) * 255, speckle)
    groove_seeds = _open_seeds((smoothed <= groove_level).astype(np.uint8) * 255, speckle)
    groove_seeds = keep_rim_lined_seeds(
        groove_seeds,
        smoothed,
        rim_level=core_level,
        probe_px=max(1, int(config.rim_probe_px)),
    )
    span_px = int(config.valley_span_px)
    seams = narrow_valley_seeds(
        smoothed,
        span_px=span_px,
        depth=float(config.valley_depth),
    )
    if np.any(seams):
        # Pale fill between a single trace's rims is also a morphological valley.
        # Restrict seams to the bright class so that only channels inside metal
        # (grey gaps between neighbouring rims) can split, not the fill itself.
        seams = np.where((seams > 0) & (smoothed >= core_level), 255, 0).astype(np.uint8)
        seams = keep_sandwiched_valley_seeds(
            seams,
            smoothed,
            span_px=span_px,
            flank_delta=max(12.0, float(config.valley_depth) * 0.25),
        )
        seams = keep_thin_valley_components(seams, max_radius=max(1.0, float(span_px) - 0.25))
        groove_seeds = cv2.bitwise_or(groove_seeds, seams)
    weak_level = min(float(core_level), float(substrate_limit) + 8.0)
    core_seeds = _cores_covering_fill(
        smoothed,
        strong_cores=strong_cores,
        groove_seeds=groove_seeds,
        weak_level=weak_level,
        speckle=speckle,
    )
    return ConductorSeeds(
        smoothed=smoothed,
        core_seeds=core_seeds,
        groove_seeds=groove_seeds,
        metal_limit=metal_limit,
    )


def gradient_watershed_mask(gray: np.ndarray, config: GradientWatershedConfig) -> np.ndarray:
    """Grow bright metal cores until they meet gap seeds; return the metal mask."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    seeds = build_conductor_seeds(source, config)
    if seeds is None:
        return np.zeros(source.shape[:2], dtype=np.uint8)
    if not seeds.has_both_classes:
        return seeds.fallback_mask()

    markers = np.zeros(source.shape, dtype=np.int32)
    _count, core_labels = cv2.connectedComponents((seeds.core_seeds > 0).astype(np.uint8), connectivity=8)
    markers[seeds.core_seeds > 0] = core_labels[seeds.core_seeds > 0] + 1
    markers[seeds.groove_seeds > 0] = 1
    cv2.watershed(cv2.cvtColor(source, cv2.COLOR_GRAY2BGR), markers)
    return ensure_binary_mask((markers > 1).astype(np.uint8) * 255)
