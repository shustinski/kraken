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

_PRESENCE_ROBUST_SPAN = 96.0
_PRESENCE_MIN_COHERENT_FRACTION = 0.002
_PRESENCE_LOCAL_CONTRAST_SPAN_FRACTION = 0.18
_PRESENCE_MIN_LOCAL_CONTRAST = 8.0
_REFINEMENT_MIN_REGION_FRACTION = 0.01
_REFINEMENT_MAX_REGION_FRACTION = 0.40
_REFINEMENT_MIN_RIDGE_TO_TRENCH_RATIO = 1.05
_REFINEMENT_MIN_CONFIRMED_PIXELS = 5


@dataclass(frozen=True, slots=True)
class GradientWatershedConfig:
    """Seed levels are expressed as margins around the two Otsu class limits."""

    smoothing_sigma: float = 1.2
    core_margin: float = 23.0
    groove_margin: float = 19.0
    rim_probe_px: int = 2
    seed_speckle_px: int = 1
    valley_span_px: int = 5
    valley_depth: float = 65.0
    random_walker_beta: float = 90.0
    random_walker_iterations: int = 160
    graph_cut_iterations: int = 5
    reconstruction_erode_px: int = 0
    boundary_relief: float = 16.0
    boundary_background_sigma: float = 12.0


@dataclass(frozen=True, slots=True)
class MetalPresenceAnalysis:
    """Interpretable evidence used to reject frames without metallization."""

    has_metal: bool
    robust_intensity_span: float
    coherent_contrast_fraction: float
    largest_coherent_contrast_fraction: float
    local_contrast_limit: float


def analyze_metal_presence(
    gray: np.ndarray,
    *,
    smoothing_sigma: float = 1.0,
) -> MetalPresenceAnalysis:
    """Detect whether a frame has histogram or spatially persistent metal evidence.

    A broad robust histogram alone is sufficient evidence because weak and
    low-area conductors can still occupy very few pixels.  When the histogram
    span is small, coherent local-contrast structures provide an independent
    route to a positive decision.  Empty SEM texture has neither property.
    """
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return MetalPresenceAnalysis(False, 0.0, 0.0, 0.0, 0.0)

    smoothed = cv2.GaussianBlur(source, (0, 0), max(0.1, float(smoothing_sigma)))
    percentile_1, percentile_99 = np.percentile(smoothed, (1.0, 99.0))
    robust_span = float(percentile_99 - percentile_1)
    local_background = cv2.GaussianBlur(smoothed, (0, 0), 12.0)
    local_contrast = cv2.absdiff(smoothed, local_background)
    contrast_limit = max(
        _PRESENCE_MIN_LOCAL_CONTRAST,
        _PRESENCE_LOCAL_CONTRAST_SPAN_FRACTION * robust_span,
    )
    coherent = cv2.morphologyEx(
        (local_contrast >= contrast_limit).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        coherent,
        connectivity=8,
    )
    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if component_count > 1 else 0
    coherent_fraction = float(np.mean(coherent > 0))
    largest_fraction = largest_area / float(source.size)
    has_histogram_evidence = robust_span >= _PRESENCE_ROBUST_SPAN
    has_coherent_evidence = largest_fraction >= _PRESENCE_MIN_COHERENT_FRACTION
    return MetalPresenceAnalysis(
        has_metal=has_histogram_evidence or has_coherent_evidence,
        robust_intensity_span=robust_span,
        coherent_contrast_fraction=coherent_fraction,
        largest_coherent_contrast_fraction=largest_fraction,
        local_contrast_limit=float(contrast_limit),
    )


def clamped_gradient_watershed_config(
    *,
    smoothing_sigma: float = 1.2,
    core_margin: float = 23.0,
    groove_margin: float = 19.0,
    rim_probe_px: int = 2,
    seed_speckle_px: int = 1,
    valley_span_px: int = 5,
    valley_depth: float = 65.0,
    random_walker_beta: float = 90.0,
    random_walker_iterations: int = 160,
    graph_cut_iterations: int = 5,
    reconstruction_erode_px: int = 0,
    boundary_relief: float = 16.0,
    boundary_background_sigma: float = 12.0,
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
        boundary_relief=max(1.0, min(80.0, float(boundary_relief))),
        boundary_background_sigma=max(2.0, min(60.0, float(boundary_background_sigma))),
    )


def gradient_watershed_config_from_object(source: object) -> GradientWatershedConfig:
    return clamped_gradient_watershed_config(
        smoothing_sigma=float(getattr(source, "watershed_smoothing_sigma", 1.2) or 1.2),
        core_margin=float(getattr(source, "watershed_core_margin", 23.0) or 0.0),
        groove_margin=float(getattr(source, "watershed_groove_margin", 19.0) or 0.0),
        rim_probe_px=int(getattr(source, "watershed_rim_probe_px", 2) or 1),
        seed_speckle_px=int(getattr(source, "watershed_seed_speckle_px", 1) or 0),
        valley_span_px=int(getattr(source, "watershed_valley_span_px", 5) or 0),
        valley_depth=float(getattr(source, "watershed_valley_depth", 65.0) or 0.0),
        random_walker_beta=float(getattr(source, "random_walker_beta", 90.0) or 90.0),
        random_walker_iterations=int(getattr(source, "random_walker_iterations", 160) or 160),
        graph_cut_iterations=int(getattr(source, "graph_cut_iterations", 5) or 5),
        reconstruction_erode_px=int(getattr(source, "reconstruction_erode_px", 0) or 0),
        boundary_relief=float(getattr(source, "boundary_relief", 16.0) or 16.0),
        boundary_background_sigma=float(getattr(source, "boundary_background_sigma", 12.0) or 12.0),
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
    """Keep enclosed substrate seeds only when a true bright rim lines them.

    Component area and a mean ring intensity are unreliable: a large dark
    conductor centre can satisfy both.  The upper-class rim level makes the
    marker conditional on actual boundary evidence.  At the FOV boundary only
    thin dark regions remain seeds; a broad dark component may be the centre of
    a conductor that continues beyond the frame.
    """
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    if count <= 1:
        return seeds
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * probe_px + 1, 2 * probe_px + 1))
    grown_near = cv2.dilate(labels.astype(np.float32), kernel).astype(np.int32)
    near_ring = (grown_near > 0) & (labels == 0)
    if not np.any(near_ring):
        return seeds
    near_maxima = np.zeros(count, dtype=np.uint8)
    np.maximum.at(near_maxima, grown_near[near_ring], smoothed[near_ring])
    keep = near_maxima >= float(rim_level)
    distance = cv2.distanceTransform((seeds > 0).astype(np.uint8), cv2.DIST_L2, 3)
    max_radius = np.zeros(count, dtype=np.float32)
    np.maximum.at(max_radius, labels.ravel(), distance.ravel())
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    keep[border_labels] |= max_radius[border_labels] <= 3.0 * float(probe_px)
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def keep_sandwiched_valley_seeds(
    seams: np.ndarray,
    smoothed: np.ndarray,
    *,
    span_px: int,
    flank_delta: float,
    support_level: float | None = None,
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
    if support_level is not None:
        outer_left = np.roll(center, 2 * span, axis=1)
        outer_right = np.roll(center, -2 * span, axis=1)
        outer_up = np.roll(center, 2 * span, axis=0)
        outer_down = np.roll(center, -2 * span, axis=0)
        support = np.int16(max(0.0, float(support_level)))
        horizontal &= (outer_left >= support) & (outer_right >= support)
        vertical &= (outer_up >= support) & (outer_down >= support)
        horizontal[:, : 2 * span] = False
        horizontal[:, -2 * span :] = False
        vertical[: 2 * span, :] = False
        vertical[-2 * span :, :] = False
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


def _estimate_noise_sigma(smoothed: np.ndarray) -> float:
    source = smoothed.astype(np.float32)
    residual = np.abs(source - cv2.GaussianBlur(source, (0, 0), 1.0))
    return 1.4826 * float(np.median(residual))


def _isolated_weak_core_seeds(
    smoothed: np.ndarray,
    *,
    weak_level: float,
    min_contrast: float,
    max_standard_deviation: float,
    probe_px: int,
) -> np.ndarray:
    """Recover bounded uniform traces that have no high-intensity rim seed."""
    weak = (smoothed >= float(weak_level)).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(weak, connectivity=8)
    if count <= 1:
        return np.zeros(smoothed.shape, dtype=np.uint8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    sums = np.bincount(
        labels.ravel(),
        weights=smoothed.ravel().astype(np.float64),
        minlength=count,
    )
    inner_means = np.divide(sums, np.maximum(areas, 1.0))
    squared_sums = np.bincount(
        labels.ravel(),
        weights=np.square(smoothed.ravel().astype(np.float64)),
        minlength=count,
    )
    variances = np.maximum(0.0, np.divide(squared_sums, np.maximum(areas, 1.0)) - inner_means**2)
    standard_deviations = np.sqrt(variances)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * probe_px + 1, 2 * probe_px + 1),
    )
    grown = cv2.dilate(labels.astype(np.float32), kernel).astype(np.int32)
    ring = (grown > 0) & (labels == 0)
    ring_labels = grown[ring]
    ring_counts = np.bincount(ring_labels, minlength=count).astype(np.float64)
    ring_sums = np.bincount(
        ring_labels,
        weights=smoothed[ring].astype(np.float64),
        minlength=count,
    )
    ring_means = np.divide(ring_sums, np.maximum(ring_counts, 1.0))
    keep = (
        (areas >= float((2 * probe_px) ** 2))
        & (areas <= 0.25 * float(smoothed.size))
        & (stats[:, cv2.CC_STAT_WIDTH] >= 2 * probe_px)
        & (stats[:, cv2.CC_STAT_HEIGHT] >= 2 * probe_px)
        & (ring_counts > 0)
        & (inner_means >= ring_means + float(min_contrast))
        & (standard_deviations <= float(max_standard_deviation))
    )
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _coherent_axis_lines(binary: np.ndarray, *, length_px: int) -> np.ndarray:
    """Retain persistent horizontal/vertical evidence while rejecting texture."""
    length = max(3, int(length_px) | 1)
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (length, 3)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, length)),
    )
    return cv2.bitwise_or(horizontal, vertical)


def _local_metal_core_seeds(
    smoothed: np.ndarray,
    *,
    substrate_limit: float,
    metal_limit: float,
    rim_level: float,
    config: GradientWatershedConfig,
) -> np.ndarray:
    """Fuse high-confidence rim, local-contrast, and bounded weak-region evidence."""
    source = smoothed.astype(np.float32)
    noise_sigma = _estimate_noise_sigma(smoothed)
    evidence_threshold = max(2.0 * float(config.core_margin), 5.0 * noise_sigma)

    probe = max(1, int(config.rim_probe_px))
    local_contrast = np.zeros(smoothed.shape, dtype=np.float32)
    for scale in sorted({probe, 2 * probe, 4 * probe}):
        local_background = cv2.GaussianBlur(source, (0, 0), float(scale))
        local_contrast = np.maximum(local_contrast, source - local_background)

    weak_level = float(substrate_limit) + float(config.groove_margin)
    broad = np.where(
        (smoothed >= float(metal_limit)) & (local_contrast >= evidence_threshold),
        255,
        0,
    ).astype(np.uint8)
    broad = _open_seeds(
        broad,
        max(int(config.seed_speckle_px), probe // 2),
    )

    isolated = _isolated_weak_core_seeds(
        smoothed,
        weak_level=weak_level,
        min_contrast=max(float(config.core_margin), 3.0 * noise_sigma),
        max_standard_deviation=max(float(config.core_margin), 2.0 * noise_sigma),
        probe_px=probe,
    )
    isolated = _open_seeds(isolated, int(config.seed_speckle_px))

    bright_candidates = np.where(
        smoothed >= max(float(metal_limit), float(rim_level)),
        255,
        0,
    ).astype(np.uint8)
    bright = cv2.bitwise_or(
        _open_seeds(bright_candidates, int(config.seed_speckle_px)),
        _coherent_axis_lines(
            bright_candidates,
            length_px=2 * int(config.seed_speckle_px) + 1,
        ),
    )
    return cv2.bitwise_or(bright, cv2.bitwise_or(broad, isolated))


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


def selective_conductor_recovery(
    confirmed_mask: np.ndarray,
    seeds: ConductorSeeds,
    config: GradientWatershedConfig,
) -> np.ndarray:
    """Fill only suspicious regions supported by coherent conductor boundaries.

    Wide, low-texture conductor interiors can be unlabeled by both seed classes
    and then assigned to the substrate by watershed.  Long ridge/trench pairs
    partition those uncertain pixels into local regions.  A region is recovered
    only when it already contains confirmed metal and its boundary is lined more
    strongly by the metal-side bright ridge than by the substrate-side trench.
    Trusted groove seeds remain hard barriers after recovery.
    """
    confirmed = ensure_binary_mask(confirmed_mask)
    probe = max(1, int(config.rim_probe_px))
    relief_background = cv2.GaussianBlur(
        seeds.smoothed,
        (0, 0),
        max(2.0, float(config.boundary_background_sigma)),
    )
    relief = seeds.smoothed.astype(np.float32) - relief_background.astype(np.float32)
    relief_limit = max(1.0, float(config.boundary_relief))
    line_length = 4 * probe + 7
    ridge = _coherent_axis_lines(
        np.where(relief >= relief_limit, 255, 0).astype(np.uint8),
        length_px=line_length,
    )
    trench = _coherent_axis_lines(
        np.where(relief <= -relief_limit, 255, 0).astype(np.uint8),
        length_px=line_length,
    )
    if not np.any(ridge) or not np.any(trench):
        return confirmed

    wall = cv2.bitwise_or(ridge, trench)
    seal_radius = max(2, probe // 2)
    wall = cv2.morphologyEx(
        wall,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * seal_radius + 1, 2 * seal_radius + 1),
        ),
    )
    uncertain = wall == 0
    region_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        uncertain.astype(np.uint8),
        connectivity=4,
    )
    if region_count <= 1:
        return confirmed

    contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(2, probe - 2) + 1, 2 * max(2, probe - 2) + 1),
    )
    ridge_contact = np.bincount(
        labels[(cv2.dilate(ridge, contact_kernel) > 0) & uncertain],
        minlength=region_count,
    )
    trench_contact = np.bincount(
        labels[(cv2.dilate(trench, contact_kernel) > 0) & uncertain],
        minlength=region_count,
    )
    confirmed_hits = np.bincount(
        labels.ravel(),
        weights=(confirmed > 0).ravel(),
        minlength=region_count,
    )
    min_region_area = max(
        60,
        (2 * probe) ** 2,
        round(_REFINEMENT_MIN_REGION_FRACTION * confirmed.size),
    )
    max_local_region_area = _REFINEMENT_MAX_REGION_FRACTION * float(confirmed.size)
    keep = (
        (stats[:, cv2.CC_STAT_AREA] >= min_region_area)
        & (stats[:, cv2.CC_STAT_AREA] <= max_local_region_area)
        & (ridge_contact > _REFINEMENT_MIN_RIDGE_TO_TRENCH_RATIO * trench_contact)
        & (confirmed_hits >= _REFINEMENT_MIN_CONFIRMED_PIXELS)
    )
    keep[0] = False
    recovered = np.where((confirmed > 0) | keep[labels], 255, 0).astype(np.uint8)
    recovered[seeds.groove_seeds > 0] = 0
    return ensure_binary_mask(recovered)


def build_conductor_seeds(
    gray: np.ndarray,
    config: GradientWatershedConfig,
    *,
    check_presence: bool = True,
) -> ConductorSeeds | None:
    """Build rim-lit metal cores and gap seeds, or None for an empty frame."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return None

    if check_presence:
        presence = analyze_metal_presence(
            source,
            smoothing_sigma=float(config.smoothing_sigma),
        )
        if not presence.has_metal:
            return None

    smoothed = cv2.GaussianBlur(source, (0, 0), max(0.1, float(config.smoothing_sigma)))
    substrate_limit, metal_limit = intensity_class_limits(smoothed)
    upper_values = smoothed[smoothed > metal_limit]
    rim_level = _otsu_level(upper_values) if upper_values.size else metal_limit
    groove_level = min(
        substrate_limit + 0.5 * float(config.groove_margin),
        metal_limit - 4.0,
    )

    speckle = max(0, int(config.seed_speckle_px))
    groove_seeds = _open_seeds((smoothed <= groove_level).astype(np.uint8) * 255, speckle)
    groove_seeds = keep_rim_lined_seeds(
        groove_seeds,
        smoothed,
        rim_level=rim_level,
        probe_px=max(1, int(config.rim_probe_px)),
    )
    span_px = int(config.valley_span_px)
    seams = narrow_valley_seeds(
        smoothed,
        span_px=span_px,
        depth=float(config.valley_depth),
    )
    if np.any(seams):
        seams = keep_sandwiched_valley_seeds(
            seams,
            smoothed,
            span_px=span_px,
            flank_delta=max(12.0, float(config.valley_depth) * 0.25),
            support_level=substrate_limit + 0.5 * float(config.groove_margin),
        )
        seams = keep_thin_valley_components(seams, max_radius=max(1.0, float(span_px) - 0.25))
        groove_seeds = cv2.bitwise_or(groove_seeds, seams)
    core_seeds = _local_metal_core_seeds(
        smoothed,
        substrate_limit=substrate_limit,
        metal_limit=metal_limit,
        rim_level=rim_level,
        config=config,
    )
    core_seeds = cv2.bitwise_and(core_seeds, cv2.bitwise_not(groove_seeds))
    return ConductorSeeds(
        smoothed=smoothed,
        core_seeds=core_seeds,
        groove_seeds=groove_seeds,
        metal_limit=metal_limit,
    )


def gradient_watershed_mask(
    gray: np.ndarray,
    config: GradientWatershedConfig,
    *,
    check_presence: bool = True,
    refine: bool = True,
) -> np.ndarray:
    """Grow bright metal cores until they meet gap seeds; return the metal mask."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return np.zeros(source.shape[:2], dtype=np.uint8)

    seeds = build_conductor_seeds(source, config, check_presence=check_presence)
    if seeds is None:
        return np.zeros(source.shape[:2], dtype=np.uint8)
    if not seeds.has_both_classes:
        return seeds.fallback_mask()

    confirmed = gradient_watershed_mask_from_seeds(source, seeds)
    if not refine:
        return confirmed
    return selective_conductor_recovery(confirmed, seeds, config)


def gradient_watershed_mask_from_seeds(
    gray: np.ndarray,
    seeds: ConductorSeeds,
) -> np.ndarray:
    """Run the baseline watershed with already-built hard seed classes."""
    source = ensure_uint8(gray)
    if source.ndim != 2 or source.size == 0:
        return np.zeros(source.shape[:2], dtype=np.uint8)
    if not seeds.has_both_classes:
        return seeds.fallback_mask()

    markers = np.zeros(source.shape, dtype=np.int32)
    # This is a binary semantic segmentation.  Independent metal seed islands
    # therefore share one label; assigning each island a watershed label creates
    # artificial one-pixel cuts where two seeds of the same conductor meet.
    markers[seeds.core_seeds > 0] = 2
    markers[seeds.groove_seeds > 0] = 1
    cv2.watershed(cv2.cvtColor(source, cv2.COLOR_GRAY2BGR), markers)
    return ensure_binary_mask((markers == 2).astype(np.uint8) * 255)
