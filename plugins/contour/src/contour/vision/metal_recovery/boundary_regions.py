"""Proof-of-concept: gradient barriers → planar regions → region labels.

Gradient magnitude answers where a physical boundary is.  Metal vs background
is decided later from grayscale, signed profiles, and core anchors.

This module is not a production strategy.  Do not register it on the detector.
It does not read ground truth, frame ids, or filenames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ...utils import ensure_binary_mask, ensure_uint8
from .gradient_watershed import GradientWatershedConfig, build_conductor_seeds
from .structural_watershed import (
    StructuralWatershedConfig,
    _StructuralFeatures,
    _extract_structural_features,
    _filter_short_components,
    _link_along_orientation,
    _non_maximum_suppress,
    _shift_no_wrap,
    clamped_structural_watershed_config,
)

LABEL_METAL = 1
LABEL_BACKGROUND = 2
LABEL_UNKNOWN = 3

_PROFILE_DISTANCES = (2.0, 3.5, 5.0)
_NMS_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1))


@dataclass(frozen=True, slots=True)
class BarrierNetworkConfig:
    hysteresis_high_percentile: float = 72.0
    hysteresis_low_percentile: float = 42.0
    min_edge_area: int = 8
    min_edge_length: int = 6
    gap_link_px: int = 2
    collapse_parallel_offset_px: int = 2


@dataclass(frozen=True, slots=True)
class RegionLabelConfig:
    core_anchor_pixels: int = 16
    core_anchor_fraction: float = 0.01
    substrate_bg_fraction: float = 0.12
    metal_score_needed: int = 3
    background_score_needed: int = 3


@dataclass(frozen=True, slots=True)
class AdjacencyRecord:
    region_a: int
    region_b: int
    length_px: int
    median_boundary_response: float
    max_boundary_response: float
    orientation_circular_mean: float
    orientation_coherence: float
    side_a_intensity: float
    side_b_intensity: float
    signed_transition: float
    profile_confidence: float


@dataclass(frozen=True, slots=True)
class RegionRecord:
    region_id: int
    area: int
    median_intensity: float
    p10_intensity: float
    p90_intensity: float
    intensity_std: float
    local_contrast_median: float
    core_pixels: int
    core_fraction: float
    substrate_pixels: int
    substrate_fraction: float
    ridge_response_mean: float
    neighbor_median_intensity: float
    intensity_vs_neighbors: float
    core_fraction_vs_neighbors: float


@dataclass
class BarrierNetwork:
    magnitude: np.ndarray
    nms: np.ndarray
    nms_collapsed: np.ndarray
    strong: np.ndarray
    weak: np.ndarray
    linked: np.ndarray
    barrier: np.ndarray


@dataclass
class BoundaryRegionResult:
    barrier_network: BarrierNetwork
    region_ids: np.ndarray
    region_count: int
    regions: list[RegionRecord]
    adjacencies: list[AdjacencyRecord]
    region_labels: np.ndarray
    metal_mask: np.ndarray
    core_seeds: np.ndarray
    substrate_seeds: np.ndarray
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


def run_boundary_region_poc(
    gray: np.ndarray,
    *,
    structural_config: StructuralWatershedConfig | None = None,
    watershed_config: GradientWatershedConfig | None = None,
    barrier_config: BarrierNetworkConfig | None = None,
    label_config: RegionLabelConfig | None = None,
) -> BoundaryRegionResult:
    source = ensure_uint8(gray)
    st_config = structural_config or clamped_structural_watershed_config(variant="s7")
    ws_config = watershed_config or GradientWatershedConfig()
    b_config = barrier_config or BarrierNetworkConfig()
    l_config = label_config or RegionLabelConfig()

    features = _extract_structural_features(source, st_config)
    seeds = build_conductor_seeds(source, ws_config, check_presence=False)
    core_seeds = (
        np.zeros(source.shape[:2], dtype=np.uint8)
        if seeds is None
        else ensure_binary_mask(seeds.core_seeds)
    )
    network = build_barrier_network(features, b_config)
    region_ids, region_count = extract_planar_regions(network.barrier)
    local_contrast = _local_contrast(features.denoised)
    substrate = _calm_substrate_seeds(features, core_seeds)
    regions, adjacencies = _region_graph(
        features,
        region_ids,
        network.barrier,
        core_seeds,
        substrate,
        local_contrast,
    )
    region_labels = label_regions(regions, adjacencies, l_config)
    metal_mask = rasterize_metal_mask(region_ids, region_labels)
    debug = _debug_images(
        features,
        network,
        region_ids,
        region_labels,
        metal_mask,
        core_seeds,
        substrate,
    )
    return BoundaryRegionResult(
        barrier_network=network,
        region_ids=region_ids,
        region_count=region_count,
        regions=regions,
        adjacencies=adjacencies,
        region_labels=region_labels,
        metal_mask=metal_mask,
        core_seeds=core_seeds,
        substrate_seeds=substrate,
        debug_images=debug,
    )


def build_barrier_network(
    features: _StructuralFeatures,
    config: BarrierNetworkConfig,
) -> BarrierNetwork:
    magnitude = features.magnitude.astype(np.float32)
    along_edge = features.structure_orientation + (np.pi * 0.5)
    nms = _non_maximum_suppress(magnitude, along_edge)
    collapsed = _collapse_parallel_duplicates(
        nms,
        magnitude,
        features.structure_orientation,
        max_offset=int(config.collapse_parallel_offset_px),
    )
    strong, weak, hyst = _hysteresis_edges(
        collapsed,
        magnitude,
        high_percentile=float(config.hysteresis_high_percentile),
        low_percentile=float(config.hysteresis_low_percentile),
    )
    cleaned = _filter_short_components(
        np.where(hyst, 255, 0).astype(np.uint8),
        min_area=int(config.min_edge_area),
        min_length=int(config.min_edge_length),
    )
    linked = _link_along_orientation(
        cleaned,
        along_edge,
        length_px=int(config.gap_link_px),
    )
    barrier = _filter_short_components(
        ensure_binary_mask(linked),
        min_area=max(4, int(config.min_edge_area) // 2),
        min_length=max(3, int(config.min_edge_length) // 2),
    )
    return BarrierNetwork(
        magnitude=magnitude,
        nms=ensure_binary_mask(nms),
        nms_collapsed=ensure_binary_mask(collapsed),
        strong=ensure_binary_mask(np.where(strong, 255, 0)),
        weak=ensure_binary_mask(np.where(weak, 255, 0)),
        linked=ensure_binary_mask(linked),
        barrier=ensure_binary_mask(barrier),
    )


def extract_planar_regions(barrier: np.ndarray) -> tuple[np.ndarray, int]:
    space = np.where(ensure_binary_mask(barrier) == 0, 255, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(space, connectivity=4)
    return labels.astype(np.int32), int(max(0, count - 1))


def label_regions(
    regions: list[RegionRecord],
    adjacencies: list[AdjacencyRecord],
    config: RegionLabelConfig,
) -> np.ndarray:
    if not regions:
        return np.zeros(0, dtype=np.int32)
    max_id = max(region.region_id for region in regions)
    labels = np.full(max_id + 1, LABEL_UNKNOWN, dtype=np.int32)
    labels[0] = 0
    scores = {
        region.region_id: _evidence_scores(region, config) for region in regions
    }
    for region in regions:
        metal_score, background_score = scores[region.region_id]
        if metal_score >= int(config.metal_score_needed) and metal_score > background_score:
            labels[region.region_id] = LABEL_METAL
        elif (
            background_score >= int(config.background_score_needed)
            and background_score > metal_score
        ):
            labels[region.region_id] = LABEL_BACKGROUND
        else:
            labels[region.region_id] = LABEL_UNKNOWN

    neighbors = _adjacency_neighbors(adjacencies)
    updated = labels.copy()
    for region in regions:
        rid = region.region_id
        if labels[rid] != LABEL_UNKNOWN:
            continue
        metal_n = 0
        background_n = 0
        for other, edge in neighbors.get(rid, ()):
            if edge.length_px < 8 or edge.profile_confidence < 0.25:
                continue
            if labels[other] == LABEL_METAL:
                metal_n += 1
            elif labels[other] == LABEL_BACKGROUND:
                background_n += 1
        metal_score, background_score = scores[rid]
        if metal_n > background_n and background_score >= 1 and metal_score == 0:
            updated[rid] = LABEL_BACKGROUND
        elif background_n > metal_n and metal_score >= 1:
            updated[rid] = LABEL_METAL
    return updated


def rasterize_metal_mask(region_ids: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    lookup = np.zeros(max(int(region_ids.max()) + 1, region_labels.size), dtype=np.uint8)
    limit = min(lookup.size, region_labels.size)
    lookup[:limit] = np.where(region_labels[:limit] == LABEL_METAL, 255, 0)
    positive = region_ids > 0
    mask = np.zeros(region_ids.shape, dtype=np.uint8)
    mask[positive] = lookup[region_ids[positive]]
    return mask


def _collapse_parallel_duplicates(
    nms: np.ndarray,
    magnitude: np.ndarray,
    across_orientation: np.ndarray,
    *,
    max_offset: int,
) -> np.ndarray:
    if max_offset <= 1:
        return nms
    mag = magnitude.astype(np.float32)
    kept = nms.copy()
    discrete = np.mod(np.round(across_orientation / (np.pi / 4.0)).astype(np.int32), 4)
    on = nms > 0
    for index, (dx, dy) in enumerate(_NMS_DIRECTIONS):
        direction = (discrete == index) & on
        if not np.any(direction):
            continue
        for offset in range(2, max_offset + 1):
            partner = _shift_no_wrap(nms, dx * offset, dy * offset) > 0
            partner_mag = _shift_no_wrap(mag, dx * offset, dy * offset)
            mid = _shift_no_wrap(mag, dx, dy) if offset == 2 else mag
            no_valley = mid >= np.minimum(mag, partner_mag) * 0.85
            weaker = direction & partner & no_valley & (mag < partner_mag)
            kept[weaker] = 0
    return kept


def _hysteresis_edges(
    nms: np.ndarray,
    magnitude: np.ndarray,
    *,
    high_percentile: float,
    low_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    on = nms > 0
    empty = np.zeros(nms.shape, dtype=bool)
    if not np.any(on):
        return empty, empty, empty
    values = magnitude[on]
    high = float(np.percentile(values, high_percentile))
    low = float(np.percentile(values, min(low_percentile, high_percentile)))
    strong = on & (magnitude >= high)
    weak = on & (magnitude >= low)
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[labels[strong]] = True
    keep[0] = False
    return strong, weak, keep[labels]


def _local_contrast(intensity: np.ndarray) -> np.ndarray:
    source = intensity.astype(np.float32)
    background = cv2.GaussianBlur(source, (0, 0), 6.0)
    return np.abs(source - background)


def _calm_substrate_seeds(features: _StructuralFeatures, core_seeds: np.ndarray) -> np.ndarray:
    intensity = features.denoised.astype(np.float32)
    dark = intensity <= float(np.percentile(intensity, 30.0))
    calm = features.magnitude <= float(np.percentile(features.magnitude, 35.0))
    away_from_cores = cv2.dilate(ensure_binary_mask(core_seeds), np.ones((15, 15), np.uint8)) == 0
    seeds = np.where(dark & calm & away_from_cores, 255, 0).astype(np.uint8)
    return _filter_short_components(seeds, min_area=32, min_length=8)


def _region_graph(
    features: _StructuralFeatures,
    region_ids: np.ndarray,
    barrier: np.ndarray,
    core_seeds: np.ndarray,
    substrate: np.ndarray,
    local_contrast: np.ndarray,
) -> tuple[list[RegionRecord], list[AdjacencyRecord]]:
    intensity = features.denoised.astype(np.float32)
    ridge = features.ridge_response.astype(np.float32)
    core = core_seeds > 0
    sub = substrate > 0
    ids = np.unique(region_ids)
    ids = ids[ids > 0]
    if ids.size == 0:
        return [], []

    max_id = int(ids.max())
    area = np.bincount(region_ids.ravel(), minlength=max_id + 1).astype(np.int32)
    core_count = np.bincount(region_ids.ravel(), weights=core.ravel().astype(np.float64), minlength=max_id + 1)
    sub_count = np.bincount(region_ids.ravel(), weights=sub.ravel().astype(np.float64), minlength=max_id + 1)
    ridge_sum = np.bincount(region_ids.ravel(), weights=ridge.ravel().astype(np.float64), minlength=max_id + 1)
    intensity_sum = np.bincount(
        region_ids.ravel(),
        weights=intensity.ravel().astype(np.float64),
        minlength=max_id + 1,
    )
    intensity_sq = np.bincount(
        region_ids.ravel(),
        weights=np.square(intensity.ravel().astype(np.float64)),
        minlength=max_id + 1,
    )
    contrast_sum = np.bincount(
        region_ids.ravel(),
        weights=local_contrast.ravel().astype(np.float64),
        minlength=max_id + 1,
    )

    percentiles = _region_intensity_percentiles(region_ids, intensity, ids)
    adjacencies = _build_adjacencies(features, region_ids, barrier, intensity)
    neighbor_intensity, neighbor_core = _neighbor_relative_stats(adjacencies, area, intensity_sum, core_count)

    records: list[RegionRecord] = []
    for rid in ids:
        pixels = int(area[rid])
        if pixels <= 0:
            continue
        mean_i = float(intensity_sum[rid] / pixels)
        var = max(0.0, float(intensity_sq[rid] / pixels) - mean_i * mean_i)
        p10, median, p90 = percentiles[int(rid)]
        core_pixels = int(core_count[rid])
        neigh_med = neighbor_intensity.get(int(rid), median)
        neigh_core = neighbor_core.get(int(rid), 0.0)
        core_frac = core_pixels / pixels
        records.append(
            RegionRecord(
                region_id=int(rid),
                area=pixels,
                median_intensity=median,
                p10_intensity=p10,
                p90_intensity=p90,
                intensity_std=float(np.sqrt(var)),
                local_contrast_median=float(contrast_sum[rid] / pixels),
                core_pixels=core_pixels,
                core_fraction=float(core_frac),
                substrate_pixels=int(sub_count[rid]),
                substrate_fraction=float(sub_count[rid] / pixels),
                ridge_response_mean=float(ridge_sum[rid] / pixels),
                neighbor_median_intensity=float(neigh_med),
                intensity_vs_neighbors=float(median - neigh_med),
                core_fraction_vs_neighbors=float(core_frac - neigh_core),
            )
        )
    return records, adjacencies


def _region_intensity_percentiles(
    region_ids: np.ndarray,
    intensity: np.ndarray,
    ids: np.ndarray,
) -> dict[int, tuple[float, float, float]]:
    order = np.argsort(region_ids.ravel(), kind="mergesort")
    sorted_ids = region_ids.ravel()[order]
    sorted_int = intensity.ravel()[order]
    starts = np.flatnonzero(np.diff(sorted_ids, prepend=sorted_ids[0] - 1))
    result: dict[int, tuple[float, float, float]] = {}
    for index, start in enumerate(starts):
        rid = int(sorted_ids[start])
        if rid <= 0:
            continue
        end = int(starts[index + 1]) if index + 1 < starts.size else int(sorted_ids.size)
        chunk = sorted_int[start:end]
        result[rid] = (
            float(np.percentile(chunk, 10.0)),
            float(np.median(chunk)),
            float(np.percentile(chunk, 90.0)),
        )
    for rid in ids:
        result.setdefault(int(rid), (0.0, 0.0, 0.0))
    return result


def _build_adjacencies(
    features: _StructuralFeatures,
    region_ids: np.ndarray,
    barrier: np.ndarray,
    intensity: np.ndarray,
) -> list[AdjacencyRecord]:
    wall = barrier > 0
    if not np.any(wall):
        return []
    rows, cols = np.nonzero(wall)
    height, width = region_ids.shape
    buckets: dict[tuple[int, int], list[int]] = {}
    neighbor_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for index, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
        seen: list[int] = []
        for dy, dx in neighbor_offsets:
            ny = row + dy
            nx = col + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width:
                continue
            rid = int(region_ids[ny, nx])
            if rid > 0 and rid not in seen:
                seen.append(rid)
        if len(seen) < 2:
            continue
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                pair = (seen[i], seen[j]) if seen[i] < seen[j] else (seen[j], seen[i])
                buckets.setdefault(pair, []).append(index)

    records: list[AdjacencyRecord] = []
    across = features.structure_orientation.astype(np.float32)
    mag = features.magnitude.astype(np.float32)
    for (id_a, id_b), indices in buckets.items():
        all_rows = rows[indices]
        all_cols = cols[indices]
        chosen = indices
        if len(chosen) > 48:
            step = max(1, len(chosen) // 48)
            chosen = chosen[::step][:48]
        sample_rows = rows[chosen]
        sample_cols = cols[chosen]
        responses = mag[all_rows, all_cols]
        angles = across[all_rows, all_cols]
        mean_cos = float(np.mean(np.cos(2.0 * angles)))
        mean_sin = float(np.mean(np.sin(2.0 * angles)))
        coherence = float(np.hypot(mean_cos, mean_sin))
        circular = 0.5 * float(np.arctan2(mean_sin, mean_cos))
        side_a, side_b, signed, confidence = _signed_profile(
            intensity,
            region_ids,
            sample_rows,
            sample_cols,
            across,
            id_a,
            id_b,
        )
        records.append(
            AdjacencyRecord(
                region_a=id_a,
                region_b=id_b,
                length_px=len(indices),
                median_boundary_response=float(np.median(responses)),
                max_boundary_response=float(np.max(responses)),
                orientation_circular_mean=circular,
                orientation_coherence=coherence,
                side_a_intensity=side_a,
                side_b_intensity=side_b,
                signed_transition=signed,
                profile_confidence=confidence,
            )
        )
    return records


def _signed_profile(
    intensity: np.ndarray,
    region_ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    across: np.ndarray,
    id_a: int,
    id_b: int,
) -> tuple[float, float, float, float]:
    height, width = intensity.shape
    samples_a: list[float] = []
    samples_b: list[float] = []
    used = 0
    for row, col in zip(rows.tolist(), cols.tolist()):
        nx = float(np.cos(across[row, col]))
        ny = float(np.sin(across[row, col]))
        oriented = _orient_normal(region_ids, row, col, nx, ny, id_a, id_b)
        if oriented is None:
            continue
        nx, ny = oriented
        values_a: list[float] = []
        values_b: list[float] = []
        for distance in _PROFILE_DISTANCES:
            ya = row - ny * distance
            xa = col - nx * distance
            yb = row + ny * distance
            xb = col + nx * distance
            if min(ya, xa, yb, xb) < 0 or ya >= height - 1 or yb >= height - 1 or xa >= width - 1 or xb >= width - 1:
                continue
            values_a.append(_sample_bilinear(intensity, ya, xa))
            values_b.append(_sample_bilinear(intensity, yb, xb))
        if not values_a or not values_b:
            continue
        samples_a.append(float(np.median(values_a)))
        samples_b.append(float(np.median(values_b)))
        used += 1
    if used == 0:
        return 0.0, 0.0, 0.0, 0.0
    side_a = float(np.median(samples_a))
    side_b = float(np.median(samples_b))
    spread = float(np.median(np.abs(np.asarray(samples_a) - np.asarray(samples_b))))
    confidence = float(min(1.0, used / max(8.0, 0.25 * float(rows.size)))) * float(
        min(1.0, spread / 8.0)
    )
    return side_a, side_b, side_b - side_a, confidence


def _orient_normal(
    region_ids: np.ndarray,
    row: int,
    col: int,
    nx: float,
    ny: float,
    id_a: int,
    id_b: int,
) -> tuple[float, float] | None:
    height, width = region_ids.shape
    probe = 2.0
    ya = int(round(row - ny * probe))
    xa = int(round(col - nx * probe))
    yb = int(round(row + ny * probe))
    xb = int(round(col + nx * probe))
    if not (0 <= ya < height and 0 <= xa < width and 0 <= yb < height and 0 <= xb < width):
        return None
    left = int(region_ids[ya, xa])
    right = int(region_ids[yb, xb])
    if left == id_a and right == id_b:
        return nx, ny
    if left == id_b and right == id_a:
        return -nx, -ny
    return nx, ny


def _sample_bilinear(values: np.ndarray, row: float, col: float) -> float:
    y0 = int(np.floor(row))
    x0 = int(np.floor(col))
    wy = row - y0
    wx = col - x0
    y1 = min(y0 + 1, values.shape[0] - 1)
    x1 = min(x0 + 1, values.shape[1] - 1)
    y0 = max(0, y0)
    x0 = max(0, x0)
    top = values[y0, x0] * (1.0 - wx) + values[y0, x1] * wx
    bottom = values[y1, x0] * (1.0 - wx) + values[y1, x1] * wx
    return float(top * (1.0 - wy) + bottom * wy)


def _neighbor_relative_stats(
    adjacencies: list[AdjacencyRecord],
    area: np.ndarray,
    intensity_sum: np.ndarray,
    core_count: np.ndarray,
) -> tuple[dict[int, float], dict[int, float]]:
    neighbor_intensity: dict[int, list[float]] = {}
    neighbor_core: dict[int, list[float]] = {}
    for edge in adjacencies:
        for src, dst in ((edge.region_a, edge.region_b), (edge.region_b, edge.region_a)):
            pixels = int(area[dst])
            if pixels <= 0:
                continue
            neighbor_intensity.setdefault(src, []).append(float(intensity_sum[dst] / pixels))
            neighbor_core.setdefault(src, []).append(float(core_count[dst] / pixels))
    return (
        {rid: float(np.median(values)) for rid, values in neighbor_intensity.items()},
        {rid: float(np.median(values)) for rid, values in neighbor_core.items()},
    )


def _evidence_scores(region: RegionRecord, config: RegionLabelConfig) -> tuple[int, int]:
    metal = 0
    background = 0
    if region.core_pixels >= int(config.core_anchor_pixels) or region.core_fraction >= float(
        config.core_anchor_fraction
    ):
        metal += 3
    if region.core_fraction_vs_neighbors > 0.005 and region.core_pixels >= 8:
        metal += 1
    if region.local_contrast_median > 6.0 and region.core_pixels >= 8:
        metal += 1
    if (
        region.substrate_fraction >= float(config.substrate_bg_fraction)
        and region.core_pixels < 8
    ):
        background += 3
    if (
        region.core_pixels == 0
        and region.substrate_fraction >= 0.04
        and region.local_contrast_median < 4.0
    ):
        background += 1
    return metal, background


def _adjacency_neighbors(
    adjacencies: list[AdjacencyRecord],
) -> dict[int, list[tuple[int, AdjacencyRecord]]]:
    neighbors: dict[int, list[tuple[int, AdjacencyRecord]]] = {}
    for edge in adjacencies:
        neighbors.setdefault(edge.region_a, []).append((edge.region_b, edge))
        neighbors.setdefault(edge.region_b, []).append((edge.region_a, edge))
    return neighbors


def _to_u8(values: np.ndarray) -> np.ndarray:
    source = values.astype(np.float32)
    low = float(np.percentile(source, 1.0))
    high = float(np.percentile(source, 99.0))
    span = max(1e-6, high - low)
    return np.clip((source - low) * (255.0 / span), 0, 255).astype(np.uint8)


def _label_color_overlay(region_ids: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*region_ids.shape, 3), dtype=np.uint8)
    lookup = np.zeros((max(int(region_ids.max()) + 1, region_labels.size), 3), dtype=np.uint8)
    limit = min(lookup.shape[0], region_labels.size)
    codes = region_labels[:limit]
    lookup[:limit][codes == LABEL_METAL] = (36, 36, 220)
    lookup[:limit][codes == LABEL_BACKGROUND] = (180, 90, 40)
    lookup[:limit][codes == LABEL_UNKNOWN] = (160, 160, 160)
    positive = region_ids > 0
    color[positive] = lookup[region_ids[positive]]
    return color


def _region_id_overlay(region_ids: np.ndarray) -> np.ndarray:
    color = np.zeros((*region_ids.shape, 3), dtype=np.uint8)
    positive = region_ids > 0
    if not np.any(positive):
        return color
    hue = ((region_ids.astype(np.int32) * 37) % 180).astype(np.uint8)
    hsv = np.zeros((*region_ids.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = np.where(positive, 200, 0).astype(np.uint8)
    hsv[:, :, 2] = np.where(positive, 220, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _debug_images(
    features: _StructuralFeatures,
    network: BarrierNetwork,
    region_ids: np.ndarray,
    region_labels: np.ndarray,
    metal_mask: np.ndarray,
    core_seeds: np.ndarray,
    substrate: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "metal_boundary_gradient_magnitude": _to_u8(network.magnitude),
        "metal_boundary_nms": network.nms,
        "metal_boundary_nms_collapsed": network.nms_collapsed,
        "metal_boundary_strong": network.strong,
        "metal_boundary_weak": network.weak,
        "metal_boundary_linked": network.linked,
        "metal_boundary_barrier": network.barrier,
        "metal_boundary_region_ids": _region_id_overlay(region_ids),
        "metal_boundary_region_labels": _label_color_overlay(region_ids, region_labels),
        "metal_boundary_metal_mask": metal_mask,
        "metal_boundary_core_anchors": core_seeds,
        "metal_boundary_substrate_seeds": substrate,
        "metal_boundary_denoised": features.denoised.astype(np.uint8),
    }
