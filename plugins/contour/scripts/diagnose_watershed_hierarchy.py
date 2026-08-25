"""Diagnostic continuous-boundary watershed hierarchy for real SEM frames.

This is an isolated experiment, not a production segmentation strategy.  The
inference half sees only the preprocessed SEM frame and builds:

    structural gradient -> minima watershed -> RAG -> merge hierarchy

Ground truth is used only after the complete merge sequence has been built, to
measure geometric partition quality at predetermined hierarchy levels.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from contour.vision.metal_recovery.structural_watershed import (  # noqa: E402
    _extract_structural_features,
    _label_overlay,
    _to_u8,
    clamped_structural_watershed_config,
)
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    EVALUATION_BORDER_CROP_PX,
    REAL_DATASET_ROOT,
    _boundary,
    build_real_benchmark_cases,
    crop_evaluation_region,
    relabel_connected_components,
)
from scripts.benchmark_structural_watershed import remap_positive_ids  # noqa: E402
from scripts.diagnose_boundary_regions import (  # noqa: E402
    _adjacent_gt_separation,
    _adjacent_label_pairs,
    _gt_fragmentation,
    _id_overlap_counts,
    _region_purity,
)

DEFAULT_FRAMES = ("3242",)
CROP_PX = EVALUATION_BORDER_CROP_PX
OUTPUT_JSON = PLUGIN_ROOT / "benchmarks" / "diagnose_watershed_hierarchy.json"
OUTPUT_ROOT = PLUGIN_ROOT / "benchmarks" / "structural_debug" / "watershed_hierarchy"
HISTOGRAM_BINS = 64
LEVEL_FRACTIONS = (
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.78,
    0.84,
    0.89,
    0.93,
    0.935,
    0.94,
    0.945,
    0.95,
    0.955,
    0.96,
    0.965,
    0.97,
    0.975,
    0.985,
    0.992,
    0.996,
)


@dataclass(frozen=True, slots=True)
class HierarchyConfig:
    terrain_smoothing_sigma: float = 1.4
    minima_window_px: int = 17
    histogram_bins: int = HISTOGRAM_BINS


@dataclass(slots=True)
class BoundaryStats:
    histogram: np.ndarray

    @property
    def length(self) -> int:
        return int(self.histogram.sum())

    def merged(self, other: BoundaryStats) -> BoundaryStats:
        return BoundaryStats(self.histogram + other.histogram)


@dataclass(frozen=True, slots=True)
class BoundarySummary:
    length: int
    mean: float
    p10: float
    p50: float
    p90: float
    maximum: float
    strong_fraction: float
    reliability: float
    score: float


@dataclass(frozen=True, slots=True)
class MergeRecord:
    region_a: int
    region_b: int
    score: float
    raw_score: float
    boundary: BoundarySummary


@dataclass(slots=True)
class InferenceResult:
    magnitude: np.ndarray
    normalized_magnitude: np.ndarray
    initial_labels: np.ndarray
    initial_boundaries: np.ndarray
    initial_basin_count: int
    initial_rag: dict[tuple[int, int], BoundaryStats]
    merge_records: list[MergeRecord]
    runtime_ms: dict[str, float]


@dataclass(slots=True)
class DisjointSet:
    parent: np.ndarray
    size: np.ndarray

    @classmethod
    def create(cls, count: int) -> DisjointSet:
        return cls(np.arange(count + 1, dtype=np.int32), np.ones(count + 1, dtype=np.int32))

    def find(self, value: int) -> int:
        root = int(value)
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(value) != root:
            next_value = int(self.parent[value])
            self.parent[value] = root
            value = next_value
        return root

    def union_keep_a(self, region_a: int, region_b: int) -> tuple[int, int]:
        root_a = self.find(region_a)
        root_b = self.find(region_b)
        if root_a == root_b:
            return root_a, root_b
        if int(self.size[root_b]) > int(self.size[root_a]):
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        return root_a, root_b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs="+", default=list(DEFAULT_FRAMES))
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()

    cases = {case.name: case for case in build_real_benchmark_cases(REAL_DATASET_ROOT)}
    missing = [name for name in args.frames if name not in cases]
    if missing:
        raise ValueError(f"Unknown real benchmark frames: {missing}")

    report: dict[str, object] = {
        "experiment": "continuous structural gradient -> watershed oversegmentation -> dynamic RAG hierarchy",
        "production_modified": False,
        "gt_during_inference": False,
        "evaluation_crop_px": CROP_PX,
        "config": {
            "terrain_smoothing_sigma": HierarchyConfig().terrain_smoothing_sigma,
            "minima_window_px": HierarchyConfig().minima_window_px,
            "histogram_bins": HierarchyConfig().histogram_bins,
            "hierarchy_level_merge_fractions": LEVEL_FRACTIONS,
        },
        "merge_score": {
            "definition": (
                "reliability * (0.50*p50 + 0.30*p90 + 0.20*strong_fraction); "
                "reliability=min(1,boundary_length/24); strong means normalized gradient >= 0.60"
            ),
            "normalization": "gradient / frame p99.5 gradient, clipped to [0,1]",
            "rag_update": "histograms of newly shared boundaries are summed and score is recomputed after every merge",
        },
        "frames": {},
    }
    for frame_id in args.frames:
        case = cases[frame_id]
        print(f"{frame_id}: building inference hierarchy", flush=True)
        inference = build_hierarchy(case.image, HierarchyConfig())
        print(
            f"{frame_id}: {inference.initial_basin_count} basins, "
            f"{len(inference.initial_rag)} RAG edges, {len(inference.merge_records)} merges",
            flush=True,
        )
        metrics = evaluate_hierarchy(frame_id, case.labels, inference)
        report["frames"][frame_id] = metrics
        save_debug(frame_id, case.image, case.labels, inference, metrics)
        print(json.dumps(_compact_console_metrics(metrics), indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def build_hierarchy(gray: np.ndarray, config: HierarchyConfig) -> InferenceResult:
    started = perf_counter()
    features = _extract_structural_features(gray, clamped_structural_watershed_config(variant="s7"))
    magnitude = features.magnitude.astype(np.float32, copy=False)
    scale = max(float(np.percentile(magnitude, 99.5)), 1e-6)
    normalized = np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)
    feature_ms = (perf_counter() - started) * 1000.0

    watershed_started = perf_counter()
    terrain = cv2.GaussianBlur(
        normalized,
        (0, 0),
        max(0.1, float(config.terrain_smoothing_sigma)),
    )
    markers = controlled_minima_markers(terrain, int(config.minima_window_px))
    terrain_u8 = np.round(terrain * 255.0).astype(np.uint8)
    watershed_labels = markers.copy()
    cv2.watershed(cv2.cvtColor(terrain_u8, cv2.COLOR_GRAY2BGR), watershed_labels)
    labels = fill_watershed_lines(watershed_labels)
    labels = remap_positive_ids(labels)
    basin_count = int(labels.max())
    boundaries = label_boundaries(labels)
    watershed_ms = (perf_counter() - watershed_started) * 1000.0

    rag_started = perf_counter()
    rag = build_rag(labels, normalized, int(config.histogram_bins))
    rag_ms = (perf_counter() - rag_started) * 1000.0

    merge_started = perf_counter()
    merges = agglomerate_rag(basin_count, rag)
    merge_ms = (perf_counter() - merge_started) * 1000.0
    return InferenceResult(
        magnitude=magnitude,
        normalized_magnitude=normalized,
        initial_labels=labels,
        initial_boundaries=boundaries,
        initial_basin_count=basin_count,
        initial_rag=rag,
        merge_records=merges,
        runtime_ms={
            "features": feature_ms,
            "watershed": watershed_ms,
            "rag": rag_ms,
            "hierarchical_merge": merge_ms,
            "total_inference": (perf_counter() - started) * 1000.0,
        },
    )


def controlled_minima_markers(terrain: np.ndarray, window_px: int) -> np.ndarray:
    window = max(3, int(window_px)) | 1
    local_minimum = terrain <= cv2.erode(terrain, np.ones((window, window), np.uint8)) + 1e-7
    count, components, stats, _centroids = cv2.connectedComponentsWithStats(
        local_minimum.astype(np.uint8), connectivity=8
    )
    markers = np.zeros(terrain.shape, dtype=np.int32)
    marker_id = 0
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component = components[y : y + height, x : x + width] == component_id
        values = terrain[y : y + height, x : x + width]
        masked = np.where(component, values, np.inf)
        local_index = int(np.argmin(masked))
        local_y, local_x = np.unravel_index(local_index, masked.shape)
        marker_id += 1
        markers[y + local_y, x + local_x] = marker_id
    if marker_id == 0:
        y, x = np.unravel_index(int(np.argmin(terrain)), terrain.shape)
        markers[y, x] = 1
    return markers


def fill_watershed_lines(labels: np.ndarray) -> np.ndarray:
    filled = np.where(labels > 0, labels, 0).astype(np.int32)
    missing = filled == 0
    for _iteration in range(4):
        if not np.any(missing):
            break
        candidates = np.zeros_like(filled)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                source_y = slice(max(0, -dy), filled.shape[0] - max(0, dy))
                source_x = slice(max(0, -dx), filled.shape[1] - max(0, dx))
                target_y = slice(max(0, dy), filled.shape[0] - max(0, -dy))
                target_x = slice(max(0, dx), filled.shape[1] - max(0, -dx))
                candidates[target_y, target_x] = np.maximum(candidates[target_y, target_x], filled[source_y, source_x])
        assign = missing & (candidates > 0)
        filled[assign] = candidates[assign]
        missing = filled == 0
    if np.any(missing):
        raise RuntimeError("Watershed left pixels farther than four pixels from a basin")
    return filled


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=np.uint8)
    vertical = labels[:-1, :] != labels[1:, :]
    horizontal = labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :][vertical] = 255
    boundary[1:, :][vertical] = 255
    boundary[:, :-1][horizontal] = 255
    boundary[:, 1:][horizontal] = 255
    return boundary


def build_rag(
    labels: np.ndarray,
    normalized_magnitude: np.ndarray,
    histogram_bins: int,
) -> dict[tuple[int, int], BoundaryStats]:
    rag: dict[tuple[int, int], BoundaryStats] = {}
    _accumulate_direction(
        rag,
        labels[:, :-1],
        labels[:, 1:],
        normalized_magnitude[:, :-1],
        normalized_magnitude[:, 1:],
        histogram_bins,
    )
    _accumulate_direction(
        rag,
        labels[:-1, :],
        labels[1:, :],
        normalized_magnitude[:-1, :],
        normalized_magnitude[1:, :],
        histogram_bins,
    )
    return rag


def _accumulate_direction(
    rag: dict[tuple[int, int], BoundaryStats],
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    magnitude_a: np.ndarray,
    magnitude_b: np.ndarray,
    histogram_bins: int,
) -> None:
    different = labels_a != labels_b
    if not np.any(different):
        return
    first = np.minimum(labels_a[different], labels_b[different]).astype(np.int64)
    second = np.maximum(labels_a[different], labels_b[different]).astype(np.int64)
    stride = int(max(labels_a.max(), labels_b.max())) + 1
    packed = first * stride + second
    strength = np.maximum(magnitude_a[different], magnitude_b[different])
    bins = np.minimum((strength * histogram_bins).astype(np.int32), histogram_bins - 1)
    keys, inverse = np.unique(packed, return_inverse=True)
    histogram = np.zeros((keys.size, histogram_bins), dtype=np.int64)
    np.add.at(histogram, (inverse, bins), 1)
    for index, key in enumerate(keys.tolist()):
        region_a = int(key // stride)
        region_b = int(key % stride)
        pair = (region_a, region_b)
        stats = BoundaryStats(histogram[index])
        existing = rag.get(pair)
        rag[pair] = stats if existing is None else existing.merged(stats)


def summarize_boundary(stats: BoundaryStats) -> BoundarySummary:
    histogram = stats.histogram.astype(np.float64, copy=False)
    length = int(histogram.sum())
    if length <= 0:
        return BoundarySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    centers = (np.arange(histogram.size, dtype=np.float64) + 0.5) / histogram.size
    cumulative = np.cumsum(histogram)

    def quantile(fraction: float) -> float:
        index = int(np.searchsorted(cumulative, fraction * length, side="left"))
        return float(centers[min(index, centers.size - 1)])

    mean = float(np.dot(histogram, centers) / length)
    p10 = quantile(0.10)
    p50 = quantile(0.50)
    p90 = quantile(0.90)
    maximum = float(centers[int(np.flatnonzero(histogram)[-1])])
    strong_start = int(np.floor(0.60 * histogram.size))
    strong_fraction = float(histogram[strong_start:].sum() / length)
    reliability = min(1.0, length / 24.0)
    score = reliability * (0.50 * p50 + 0.30 * p90 + 0.20 * strong_fraction)
    return BoundarySummary(
        length,
        mean,
        p10,
        p50,
        p90,
        maximum,
        strong_fraction,
        reliability,
        score,
    )


def agglomerate_rag(
    basin_count: int,
    initial_rag: dict[tuple[int, int], BoundaryStats],
) -> list[MergeRecord]:
    dsu = DisjointSet.create(basin_count)
    adjacency: dict[int, dict[int, BoundaryStats]] = {region: {} for region in range(1, basin_count + 1)}
    versions: dict[tuple[int, int], int] = {}
    heap: list[tuple[float, int, int, int]] = []

    def install(region_a: int, region_b: int, stats: BoundaryStats) -> None:
        if region_a == region_b:
            return
        low, high = sorted((region_a, region_b))
        adjacency[low][high] = stats
        adjacency[high][low] = stats
        pair = (low, high)
        version = versions.get(pair, 0) + 1
        versions[pair] = version
        heapq.heappush(heap, (summarize_boundary(stats).score, low, high, version))

    for (region_a, region_b), stats in initial_rag.items():
        install(region_a, region_b, stats)

    records: list[MergeRecord] = []
    monotone_score = 0.0
    while heap and len(records) < basin_count - 1:
        raw_score, region_a, region_b, version = heapq.heappop(heap)
        root_a = dsu.find(region_a)
        root_b = dsu.find(region_b)
        if root_a != region_a or root_b != region_b or root_a == root_b:
            continue
        pair = (min(root_a, root_b), max(root_a, root_b))
        if versions.get(pair) != version:
            continue
        shared = adjacency[root_a].get(root_b)
        if shared is None:
            continue
        summary = summarize_boundary(shared)
        monotone_score = max(monotone_score, float(raw_score))
        records.append(MergeRecord(root_a, root_b, monotone_score, float(raw_score), summary))

        neighbors_a = dict(adjacency[root_a])
        neighbors_b = dict(adjacency[root_b])
        keep, removed = dsu.union_keep_a(root_a, root_b)
        keep_neighbors = neighbors_a if keep == root_a else neighbors_b
        removed_neighbors = neighbors_b if removed == root_b else neighbors_a
        all_neighbors = set(keep_neighbors) | set(removed_neighbors)
        all_neighbors.discard(root_a)
        all_neighbors.discard(root_b)

        for neighbor in tuple(adjacency[keep]):
            adjacency[neighbor].pop(keep, None)
        for neighbor in tuple(adjacency[removed]):
            adjacency[neighbor].pop(removed, None)
        adjacency[keep].clear()
        adjacency[removed].clear()
        for neighbor in all_neighbors:
            neighbor_root = dsu.find(neighbor)
            if neighbor_root in {keep, removed}:
                continue
            stats_a = keep_neighbors.get(neighbor)
            stats_b = removed_neighbors.get(neighbor)
            if stats_a is None:
                combined = stats_b
            elif stats_b is None:
                combined = stats_a
            else:
                combined = stats_a.merged(stats_b)
            if combined is not None:
                install(keep, neighbor_root, combined)

    if len(records) != basin_count - 1:
        raise RuntimeError(f"RAG hierarchy disconnected: {len(records)} merges for {basin_count} basins")
    return records


def evaluate_hierarchy(
    frame_id: str,
    gt_full: np.ndarray,
    inference: InferenceResult,
) -> dict[str, object]:
    initial_count = inference.initial_basin_count
    steps = sorted({min(initial_count - 1, round(fraction * (initial_count - 1))) for fraction in LEVEL_FRACTIONS})
    gt = relabel_connected_components(crop_evaluation_region(gt_full, CROP_PX, frame_id=frame_id))
    levels: list[dict[str, object]] = []
    for step in steps:
        labels_full = replay_partition(inference.initial_labels, inference.merge_records, step)
        labels = remap_positive_ids(crop_evaluation_region(labels_full, CROP_PX, frame_id=frame_id))
        boundary = label_boundaries(labels)
        purity = _region_purity(labels, gt)
        fragmentation = _gt_fragmentation(labels, gt)
        adjacent = _adjacent_gt_separation(labels, boundary, gt)
        level = {
            "merge_operations": step,
            "hierarchy_fraction": float(step / max(1, initial_count - 1)),
            "merge_score": 0.0 if step == 0 else float(inference.merge_records[step - 1].score),
            "predicted_regions": int(labels.max()),
            "regions_overlapping_0_gt": int(purity["regions_overlapping_0_gt"]),
            "regions_overlapping_1_gt": int(purity["regions_overlapping_1_gt"]),
            "regions_overlapping_more_than_1_gt": int(purity["regions_overlapping_more_than_1_gt"]),
            "gts_inside_multi_gt_regions": int(purity["unique_gts_inside_multi_gt_regions"]),
            "regions_per_gt_mean": float(fragmentation["mean"]),
            "regions_per_gt_median": float(fragmentation["median"]),
            "regions_per_gt_p90": float(fragmentation["p90"]),
            "gt_with_1_region": int(fragmentation["gt_with_1_region"]),
            "gt_with_more_than_1_region": int(fragmentation["gt_with_gt1_regions"]),
            "adjacent_pair_count": int(adjacent["adjacent_pair_count"]),
            "adjacent_pairs_separated": int(adjacent["disjoint_predicted_regions"]),
            "adjacent_pairs_sharing_region": int(adjacent["shared_predicted_region"]),
            "adjacent_pairs_distinct_majority": int(adjacent["distinct_majority_region"]),
        }
        if frame_id == "0175":
            level["wide_plate"] = wide_plate_stats(labels, gt)
        levels.append(level)

    pareto_indices = pareto_front(levels)
    best_index = choose_pareto_point(levels, pareto_indices)
    best = levels[best_index]
    gate = geometry_gate(best) if frame_id == "3242" else None
    representative = representative_levels(levels, best_index)
    return {
        "initial_watershed_basins_full": initial_count,
        "initial_rag_edges": len(inference.initial_rag),
        "merge_operations_total": len(inference.merge_records),
        "initial_boundary_statistics": aggregate_rag_statistics(inference.initial_rag),
        "runtime_ms": inference.runtime_ms,
        "hierarchy_levels": levels,
        "representative_levels": representative,
        "pareto_level_indices": pareto_indices,
        "best_pareto_index": best_index,
        "best_pareto_point": best,
        "geometry_gate": gate,
    }


def replay_partition(initial_labels: np.ndarray, records: list[MergeRecord], merge_count: int) -> np.ndarray:
    basin_count = int(initial_labels.max())
    dsu = DisjointSet.create(basin_count)
    for record in records[:merge_count]:
        dsu.union_keep_a(record.region_a, record.region_b)
    lookup = np.arange(basin_count + 1, dtype=np.int32)
    for region in range(1, basin_count + 1):
        lookup[region] = dsu.find(region)
    return remap_positive_ids(lookup[initial_labels])


def pareto_front(levels: list[dict[str, object]]) -> list[int]:
    result: list[int] = []
    for index, level in enumerate(levels):
        fragmentation = float(level["regions_per_gt_p90"])
        sharing = int(level["adjacent_pairs_sharing_region"])
        dominated = False
        for other_index, other in enumerate(levels):
            if other_index == index:
                continue
            other_fragmentation = float(other["regions_per_gt_p90"])
            other_sharing = int(other["adjacent_pairs_sharing_region"])
            if (
                other_fragmentation <= fragmentation
                and other_sharing <= sharing
                and (other_fragmentation < fragmentation or other_sharing < sharing)
            ):
                dominated = True
                break
        if not dominated:
            result.append(index)
    return result


def choose_pareto_point(levels: list[dict[str, object]], pareto_indices: list[int]) -> int:
    moderate = [
        index
        for index in pareto_indices
        if float(levels[index]["regions_per_gt_median"]) <= 1.0 and float(levels[index]["regions_per_gt_p90"]) <= 4.0
    ]
    candidates = moderate or pareto_indices
    return min(
        candidates,
        key=lambda index: (
            int(levels[index]["adjacent_pairs_sharing_region"]),
            int(levels[index]["gts_inside_multi_gt_regions"]),
            float(levels[index]["regions_per_gt_p90"]),
        ),
    )


def geometry_gate(level: dict[str, object]) -> dict[str, object]:
    separated = int(level["adjacent_pairs_separated"])
    median = float(level["regions_per_gt_median"])
    p90 = float(level["regions_per_gt_p90"])
    multi_gt = int(level["gts_inside_multi_gt_regions"])
    passed = separated > 240 and median <= 1.0 and p90 <= 4.0 and multi_gt < 317
    return {
        "passed": passed,
        "criteria": {
            "adjacent_pairs_separated_gt_240": separated > 240,
            "median_regions_per_gt_le_1": median <= 1.0,
            "p90_regions_per_gt_le_4": p90 <= 4.0,
            "gts_inside_multi_gt_lt_317": multi_gt < 317,
        },
        "note": "Post-hoc diagnostic gate; no criterion participates in inference or merge selection.",
    }


def wide_plate_stats(labels: np.ndarray, gt: np.ndarray) -> dict[str, object]:
    gt_ids, counts = np.unique(gt[gt > 0], return_counts=True)
    if gt_ids.size == 0:
        return {"present": False}
    plate_id = int(gt_ids[int(np.argmax(counts))])
    plate = gt == plate_id
    overlapping, overlap_counts = np.unique(labels[plate], return_counts=True)
    positive = overlapping > 0
    overlap_counts = overlap_counts[positive]
    pieces = int(np.count_nonzero(positive))
    plate_pixels = int(np.count_nonzero(plate))
    return {
        "present": True,
        "plate_pixels": plate_pixels,
        "predicted_regions_intersecting_plate": pieces,
        "largest_region_fraction_of_plate": (
            0.0 if overlap_counts.size == 0 else float(overlap_counts.max() / max(1, plate_pixels))
        ),
    }


def aggregate_rag_statistics(rag: dict[tuple[int, int], BoundaryStats]) -> dict[str, float | int]:
    summaries = [summarize_boundary(stats) for stats in rag.values()]
    if not summaries:
        return {"edge_count": 0}
    lengths = np.asarray([summary.length for summary in summaries], dtype=np.float64)
    means = np.asarray([summary.mean for summary in summaries], dtype=np.float64)
    medians = np.asarray([summary.p50 for summary in summaries], dtype=np.float64)
    p10s = np.asarray([summary.p10 for summary in summaries], dtype=np.float64)
    p90s = np.asarray([summary.p90 for summary in summaries], dtype=np.float64)
    maxima = np.asarray([summary.maximum for summary in summaries], dtype=np.float64)
    strong = np.asarray([summary.strong_fraction for summary in summaries], dtype=np.float64)
    return {
        "edge_count": len(summaries),
        "boundary_length_median": float(np.median(lengths)),
        "boundary_length_p90": float(np.percentile(lengths, 90.0)),
        "gradient_mean_median": float(np.median(means)),
        "gradient_p10_median": float(np.median(p10s)),
        "gradient_p50_median": float(np.median(medians)),
        "gradient_p90_median": float(np.median(p90s)),
        "gradient_max_median": float(np.median(maxima)),
        "strong_fraction_median": float(np.median(strong)),
    }


def representative_levels(levels: list[dict[str, object]], best_index: int) -> list[dict[str, object]]:
    indices = {0, len(levels) - 1, best_index}
    for fraction in (0.25, 0.50, 0.75):
        indices.add(min(range(len(levels)), key=lambda index: abs(index / max(1, len(levels) - 1) - fraction)))
    return [levels[index] for index in sorted(indices)]


def save_debug(
    frame_id: str,
    image: np.ndarray,
    gt_full: np.ndarray,
    inference: InferenceResult,
    metrics: dict[str, object],
) -> None:
    folder = OUTPUT_ROOT / frame_id
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "01_structural_gradient.png"), _to_u8(inference.magnitude))
    cv2.imwrite(str(folder / "02_initial_watershed_labels.png"), _label_overlay(inference.initial_labels))
    cv2.imwrite(str(folder / "03_initial_watershed_boundaries.png"), inference.initial_boundaries)
    cv2.imwrite(str(folder / "04_shared_boundary_strength.png"), shared_boundary_strength_map(inference))
    cv2.imwrite(str(folder / "05_rag_visualization.png"), rag_visualization(image, inference))

    levels = metrics["hierarchy_levels"]
    assert isinstance(levels, list)
    best_index = int(metrics["best_pareto_index"])
    selected = sorted({0, best_index, len(levels) - 1})
    for ordinal, level_index in enumerate(selected):
        level = levels[level_index]
        assert isinstance(level, dict)
        step = int(level["merge_operations"])
        labels_full = replay_partition(inference.initial_labels, inference.merge_records, step)
        tag = ("undermerged", "pareto", "overmerged")[ordinal] if len(selected) == 3 else f"level_{ordinal}"
        cv2.imwrite(str(folder / f"partition_{tag}.png"), _label_overlay(labels_full))
        save_benchmark_overlays(folder, tag, image, gt_full, labels_full)

    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def shared_boundary_strength_map(inference: InferenceResult) -> np.ndarray:
    output = np.zeros(inference.initial_labels.shape, dtype=np.uint8)
    score_by_pair = {pair: summarize_boundary(stats).score for pair, stats in inference.initial_rag.items()}
    for first, second, first_slice, second_slice in (
        (
            inference.initial_labels[:, :-1],
            inference.initial_labels[:, 1:],
            (slice(None), slice(None, -1)),
            (slice(None), slice(1, None)),
        ),
        (
            inference.initial_labels[:-1, :],
            inference.initial_labels[1:, :],
            (slice(None, -1), slice(None)),
            (slice(1, None), slice(None)),
        ),
    ):
        different = first != second
        if not np.any(different):
            continue
        low = np.minimum(first[different], second[different])
        high = np.maximum(first[different], second[different])
        values = np.fromiter(
            (round(255.0 * score_by_pair[(int(a), int(b))]) for a, b in zip(low, high, strict=True)),
            dtype=np.uint8,
            count=low.size,
        )
        view_a = output[first_slice]
        view_b = output[second_slice]
        view_a[different] = np.maximum(view_a[different], values)
        view_b[different] = np.maximum(view_b[different], values)
    return output


def rag_visualization(image: np.ndarray, inference: InferenceResult) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labels = inference.initial_labels
    count = inference.initial_basin_count
    area = np.bincount(labels.ravel(), minlength=count + 1).astype(np.float64)
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    sum_x = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=count + 1)
    sum_y = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=count + 1)
    center_x = np.divide(sum_x, np.maximum(area, 1.0))
    center_y = np.divide(sum_y, np.maximum(area, 1.0))
    ranked = sorted(
        inference.initial_rag.items(),
        key=lambda item: (summarize_boundary(item[1]).length, summarize_boundary(item[1]).score),
        reverse=True,
    )[:2000]
    for (region_a, region_b), stats in ranked:
        score = summarize_boundary(stats).score
        color = (int(255 * (1.0 - score)), 80, int(255 * score))
        cv2.line(
            canvas,
            (round(center_x[region_a]), round(center_y[region_a])),
            (round(center_x[region_b]), round(center_y[region_b])),
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_benchmark_overlays(
    folder: Path,
    tag: str,
    image_full: np.ndarray,
    gt_full: np.ndarray,
    labels_full: np.ndarray,
) -> None:
    image = crop_evaluation_region(image_full, CROP_PX)
    gt = relabel_connected_components(crop_evaluation_region(gt_full, CROP_PX))
    labels = remap_positive_ids(crop_evaluation_region(labels_full, CROP_PX))
    gt_boundary = _boundary(gt)
    predicted_boundary = label_boundaries(labels) > 0
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[gt_boundary] = (40, 200, 40)
    overlay[predicted_boundary] = (30, 30, 220)
    cv2.imwrite(str(folder / f"benchmark_{tag}_gt_boundaries.png"), overlay)

    region_to_gt = _id_overlap_counts(labels, gt)
    multi_regions = {region for region, gt_ids in region_to_gt.items() if len(gt_ids) > 1}
    multi_mask = (
        np.isin(labels, np.fromiter(multi_regions, dtype=np.int32)) if multi_regions else np.zeros(labels.shape, bool)
    )
    multi = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    multi[multi_mask] = (0, 0, 255)
    cv2.imwrite(str(folder / f"benchmark_{tag}_multi_gt_regions.png"), multi)

    gt_to_region = _id_overlap_counts(gt, labels)
    fragmented_gt = {gt_id for gt_id, region_ids in gt_to_region.items() if len(region_ids) > 1}
    fragmented_mask = (
        np.isin(gt, np.fromiter(fragmented_gt, dtype=np.int32)) if fragmented_gt else np.zeros(gt.shape, bool)
    )
    fragmented = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    fragmented[fragmented_mask] = (0, 180, 255)
    cv2.imwrite(str(folder / f"benchmark_{tag}_fragmented_gt.png"), fragmented)

    unseparated_gt_ids: set[int] = set()
    gt_to_regions = _id_overlap_counts(gt, labels)
    for gt_a, gt_b in _adjacent_label_pairs(gt, max_gap_px=8):
        if not gt_to_regions.get(gt_a, set()).isdisjoint(gt_to_regions.get(gt_b, set())):
            unseparated_gt_ids.update((gt_a, gt_b))
    unseparated_mask = (
        np.isin(gt, np.fromiter(unseparated_gt_ids, dtype=np.int32)) if unseparated_gt_ids else np.zeros(gt.shape, bool)
    )
    unseparated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    unseparated[unseparated_mask] = (180, 0, 220)
    cv2.imwrite(str(folder / f"benchmark_{tag}_adjacent_pairs_not_separated.png"), unseparated)


def _compact_console_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "initial_watershed_basins_full": metrics["initial_watershed_basins_full"],
        "initial_rag_edges": metrics["initial_rag_edges"],
        "merge_operations_total": metrics["merge_operations_total"],
        "runtime_ms": metrics["runtime_ms"],
        "best_pareto_point": metrics["best_pareto_point"],
        "geometry_gate": metrics["geometry_gate"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
