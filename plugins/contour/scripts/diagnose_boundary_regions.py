"""Benchmark the boundary→region PoC on 3242 and 0175.

Inference uses only the SEM frame.  GT is applied after crop-50 evaluation.
Production recognition and S7–S15 strategies are not modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from contour.vision.metal_recovery.boundary_regions import (  # noqa: E402
    LABEL_BACKGROUND,
    LABEL_METAL,
    LABEL_UNKNOWN,
    run_boundary_region_poc,
)
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    EVALUATION_BORDER_CROP_PX,
    REAL_DATASET_ROOT,
    _boundary,
    build_real_benchmark_cases,
    crop_evaluation_region,
    measure_segmentation,
    relabel_connected_components,
)
from scripts.benchmark_structural_watershed import remap_positive_ids  # noqa: E402

FRAMES = ("3242", "0175")
CROP_PX = EVALUATION_BORDER_CROP_PX
OUTPUT_JSON = PLUGIN_ROOT / "benchmarks" / "diagnose_boundary_regions.json"
OUTPUT_ROOT = PLUGIN_ROOT / "benchmarks" / "structural_debug" / "boundary_regions"


def main() -> int:
    cases = {case.name: case for case in build_real_benchmark_cases(REAL_DATASET_ROOT)}
    report: dict[str, object] = {
        "evaluation": {
            "full_frame": "2000x2000",
            "crop_px": CROP_PX,
            "roi": "1900x1900",
            "gt_during_inference": False,
        },
        "frames": {},
    }
    for frame_id in FRAMES:
        case = cases[frame_id]
        result = run_boundary_region_poc(case.image)
        metrics = evaluate_frame(frame_id, case.image, case.labels, result)
        report["frames"][frame_id] = metrics
        save_debug(frame_id, case.image, case.labels, result, metrics)
        print(frame_id, json.dumps(_printable(metrics), indent=2))
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    return 0


def evaluate_frame(frame_id: str, image: np.ndarray, gt_full: np.ndarray, result) -> dict[str, object]:
    gray = crop_evaluation_region(image, CROP_PX, frame_id=frame_id)
    gt = relabel_connected_components(crop_evaluation_region(gt_full, CROP_PX, frame_id=frame_id))
    barrier = crop_evaluation_region(result.barrier_network.barrier, CROP_PX, frame_id=frame_id)
    regions = remap_positive_ids(crop_evaluation_region(result.region_ids, CROP_PX, frame_id=frame_id))
    labels_lookup = result.region_labels
    metal = crop_evaluation_region(result.metal_mask, CROP_PX, frame_id=frame_id)
    cropped_ids = crop_evaluation_region(result.region_ids, CROP_PX, frame_id=frame_id)

    boundary = _boundary_quality(barrier, gt)
    purity = _region_purity(regions, gt)
    containment = _single_gt_containment(regions, gt)
    fragmentation = _gt_fragmentation(regions, gt)
    pairs = _adjacent_gt_separation(regions, barrier, gt)
    classification = _classification_benchmark(cropped_ids, labels_lookup, gt)
    seg = measure_segmentation(metal, gt, elapsed_ms=0.0)
    unknown_area = _unknown_area(cropped_ids, labels_lookup)

    payload: dict[str, object] = {
        "planar_region_count_full": int(result.region_count),
        "planar_region_count_eval": int(np.unique(regions[regions > 0]).size),
        "barrier_pixels_eval": int(np.count_nonzero(barrier)),
        "boundary_quality": boundary,
        "region_purity": purity,
        "single_gt_containment": containment,
        "gt_fragmentation": fragmentation,
        "adjacent_gt_pairs": pairs,
        "classification": classification,
        "unknown_area_fraction": unknown_area,
        "mask": {
            "iou": float(seg.iou),
            "precision": float(seg.precision),
            "recall": float(seg.recall),
        },
    }
    if frame_id == "0175":
        payload["wide_plate"] = _wide_plate_stats(regions, cropped_ids, labels_lookup, gt)
    return payload


def _boundary_quality(barrier: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    gt_boundary = _boundary(gt)
    det = barrier > 0
    out: dict[str, float] = {}
    for radius in (1, 2, 3):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        near_det = cv2.dilate(det.astype(np.uint8), kernel) > 0
        near_gt = cv2.dilate(gt_boundary.astype(np.uint8), kernel) > 0
        gt_count = int(np.count_nonzero(gt_boundary))
        det_count = int(np.count_nonzero(det))
        out[f"recall_{radius}px"] = (
            1.0 if gt_count == 0 else float(np.count_nonzero(gt_boundary & near_det) / gt_count)
        )
        out[f"precision_{radius}px"] = (
            1.0 if det_count == 0 else float(np.count_nonzero(det & near_gt) / det_count)
        )
    return out


def _region_purity(regions: np.ndarray, gt: np.ndarray) -> dict[str, object]:
    overlap = _id_overlap_counts(regions, gt)
    zero = one = many = 0
    many_ids: list[int] = []
    max_id = int(regions.max())
    area = np.bincount(regions.ravel(), minlength=max_id + 1)
    multi_details: list[dict[str, int]] = []
    swallowed: set[int] = set()
    for rid, gt_ids in overlap.items():
        count = len(gt_ids)
        if count == 0:
            zero += 1
        elif count == 1:
            one += 1
        else:
            many += 1
            many_ids.append(rid)
            swallowed.update(gt_ids)
            multi_details.append(
                {
                    "region_id": int(rid),
                    "gt_count": count,
                    "area": int(area[rid]),
                }
            )
    multi_details.sort(key=lambda item: item["gt_count"], reverse=True)
    return {
        "regions_overlapping_0_gt": zero,
        "regions_overlapping_1_gt": one,
        "regions_overlapping_more_than_1_gt": many,
        "unique_gts_inside_multi_gt_regions": len(swallowed),
        "multi_gt_regions": multi_details[:12],
        "multi_gt_region_ids_sample": many_ids[:20],
    }


def _single_gt_containment(regions: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    overlap = _id_overlap_counts(regions, gt)
    max_id = int(regions.max())
    area = np.bincount(regions.ravel(), minlength=max_id + 1)
    metal = np.bincount(
        regions.ravel(),
        weights=(gt > 0).ravel().astype(np.float64),
        minlength=max_id + 1,
    )
    fractions: list[float] = []
    for rid, gt_ids in overlap.items():
        if len(gt_ids) != 1 or int(area[rid]) <= 0:
            continue
        fractions.append(float(metal[rid] / area[rid]))
    if not fractions:
        return {"count": 0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    array = np.asarray(fractions, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
    }


def _gt_fragmentation(regions: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    overlap = _id_overlap_counts(gt, regions)
    counts = [len(region_ids) for _gt_id, region_ids in overlap.items()]
    if not counts:
        return {"median": 0.0, "p90": 0.0, "mean": 0.0, "gt_count": 0}
    array = np.asarray(counts, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "mean": float(np.mean(array)),
        "gt_count": int(array.size),
        "gt_with_1_region": int(np.count_nonzero(array == 1)),
        "gt_with_gt1_regions": int(np.count_nonzero(array > 1)),
    }


def _adjacent_gt_separation(
    regions: np.ndarray,
    barrier: np.ndarray,
    gt: np.ndarray,
) -> dict[str, object]:
    pairs = _adjacent_label_pairs(gt, max_gap_px=8)
    near_barrier = cv2.dilate((barrier > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    majority = _majority_overlap(gt, regions)
    overlap = _id_overlap_counts(gt, regions)
    distinct_majority = 0
    disjoint_sets = 0
    shared_region = 0
    for id_a, id_b in pairs:
        set_a = overlap.get(id_a, set())
        set_b = overlap.get(id_b, set())
        if set_a.isdisjoint(set_b):
            disjoint_sets += 1
        else:
            shared_region += 1
        maj_a = majority.get(id_a, 0)
        maj_b = majority.get(id_b, 0)
        if maj_a != maj_b and min(maj_a, maj_b) > 0:
            distinct_majority += 1
    coverage, gaps = _pair_separator_stats(gt, near_barrier, pairs)
    return {
        "adjacent_pair_count": len(pairs),
        "adjacency_gap_px": 8,
        "disjoint_predicted_regions": disjoint_sets,
        "shared_predicted_region": shared_region,
        "distinct_majority_region": distinct_majority,
        "separator_coverage_median": float(np.median(coverage)) if coverage.size else 0.0,
        "separator_coverage_mean": float(np.mean(coverage)) if coverage.size else 0.0,
        "max_gap_median": float(np.median(gaps)) if gaps.size else 0.0,
        "max_gap_p90": float(np.percentile(gaps, 90.0)) if gaps.size else 0.0,
        "old_merge_cases_shared_region": shared_region,
    }


def _classification_benchmark(
    region_ids: np.ndarray,
    region_labels: np.ndarray,
    gt: np.ndarray,
) -> dict[str, object]:
    max_id = int(region_ids.max())
    area = np.bincount(region_ids.ravel(), minlength=max_id + 1)
    metal_pixels = np.bincount(
        region_ids.ravel(),
        weights=(gt > 0).ravel().astype(np.float64),
        minlength=max_id + 1,
    )
    correct_metal = incorrect_metal = correct_bg = incorrect_bg = unknown = 0
    metal_area = bg_area = unknown_area = 0
    for rid in range(1, max_id + 1):
        pixels = int(area[rid])
        if pixels <= 0:
            continue
        code = int(region_labels[rid]) if rid < region_labels.size else LABEL_UNKNOWN
        metal_frac = float(metal_pixels[rid] / pixels)
        if code == LABEL_METAL:
            metal_area += pixels
            if metal_frac >= 0.5:
                correct_metal += 1
            else:
                incorrect_metal += 1
        elif code == LABEL_BACKGROUND:
            bg_area += pixels
            if metal_frac < 0.5:
                correct_bg += 1
            else:
                incorrect_bg += 1
        else:
            unknown += 1
            unknown_area += pixels
    return {
        "correctly_metal_regions": correct_metal,
        "incorrectly_metal_regions": incorrect_metal,
        "correctly_background_regions": correct_bg,
        "incorrectly_background_regions": incorrect_bg,
        "unknown_regions": unknown,
        "metal_region_area": metal_area,
        "background_region_area": bg_area,
        "unknown_region_area": unknown_area,
    }


def _unknown_area(region_ids: np.ndarray, region_labels: np.ndarray) -> float:
    lookup = np.zeros(max(int(region_ids.max()) + 1, region_labels.size), dtype=np.uint8)
    limit = min(lookup.size, region_labels.size)
    lookup[:limit] = region_labels[:limit] == LABEL_UNKNOWN
    unknown = np.zeros(region_ids.shape, dtype=bool)
    positive = region_ids > 0
    unknown[positive] = lookup[region_ids[positive]] > 0
    return float(np.count_nonzero(unknown) / max(1, unknown.size))


def _wide_plate_stats(
    regions: np.ndarray,
    region_ids_cropped: np.ndarray,
    region_labels: np.ndarray,
    gt: np.ndarray,
) -> dict[str, object]:
    gt_ids, counts = np.unique(gt[gt > 0], return_counts=True)
    if gt_ids.size == 0:
        return {"present": False}
    plate_id = int(gt_ids[int(np.argmax(counts))])
    plate = gt == plate_id
    overlapping = np.unique(region_ids_cropped[plate & (region_ids_cropped > 0)])
    area_by_region = []
    labels = []
    for rid in overlapping.tolist():
        area = int(np.count_nonzero(plate & (region_ids_cropped == rid)))
        code = int(region_labels[rid]) if 0 < rid < region_labels.size else LABEL_UNKNOWN
        area_by_region.append({"region_id": int(rid), "plate_pixels": area, "label": _label_name(code)})
        labels.append(code)
    area_by_region.sort(key=lambda item: item["plate_pixels"], reverse=True)
    primary = area_by_region[0] if area_by_region else None
    return {
        "present": True,
        "gt_plate_pixels": int(np.count_nonzero(plate)),
        "predicted_region_count_on_plate": int(overlapping.size),
        "plate_regions_over_1000px": int(sum(1 for item in area_by_region if item["plate_pixels"] >= 1000)),
        "primary_region": primary,
        "label_set": sorted({_label_name(code) for code in labels}),
        "regions": area_by_region[:12],
    }


def _id_overlap_counts(source: np.ndarray, target: np.ndarray) -> dict[int, set[int]]:
    mask = (source > 0) & (target > 0)
    if not np.any(mask):
        ids = np.unique(source[source > 0])
        return {int(sid): set() for sid in ids.tolist()}
    packed = source[mask].astype(np.int64) * (int(target.max()) + 1) + target[mask].astype(np.int64)
    unique = np.unique(packed)
    stride = int(target.max()) + 1
    mapping: dict[int, set[int]] = {int(sid): set() for sid in np.unique(source[source > 0]).tolist()}
    for value in unique.tolist():
        sid = int(value // stride)
        tid = int(value % stride)
        if sid > 0 and tid > 0:
            mapping.setdefault(sid, set()).add(tid)
    return mapping


def _majority_overlap(source: np.ndarray, target: np.ndarray) -> dict[int, int]:
    mask = (source > 0) & (target > 0)
    if not np.any(mask):
        return {}
    stride = int(target.max()) + 1
    packed = source[mask].astype(np.int64) * stride + target[mask].astype(np.int64)
    keys, counts = np.unique(packed, return_counts=True)
    best: dict[int, tuple[int, int]] = {}
    for key, count in zip(keys.tolist(), counts.tolist()):
        sid = int(key // stride)
        tid = int(key % stride)
        prev = best.get(sid)
        if prev is None or count > prev[1]:
            best[sid] = (tid, count)
    return {sid: tid for sid, (tid, _count) in best.items()}


def _adjacent_label_pairs(labels: np.ndarray, max_gap_px: int = 8) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    directions = ((0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1))
    for distance in range(1, max_gap_px + 1):
        for dy, dx in directions:
            shifted = np.zeros_like(labels)
            step_y = dy * distance
            step_x = dx * distance
            src_y0 = max(0, -step_y)
            src_y1 = labels.shape[0] - max(0, step_y)
            src_x0 = max(0, -step_x)
            src_x1 = labels.shape[1] - max(0, step_x)
            dst_y0 = max(0, step_y)
            dst_y1 = labels.shape[0] - max(0, -step_y)
            dst_x0 = max(0, step_x)
            dst_x1 = labels.shape[1] - max(0, -step_x)
            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = labels[src_y0:src_y1, src_x0:src_x1]
            touch = (labels > 0) & (shifted > 0) & (labels != shifted)
            if not np.any(touch):
                continue
            lo = np.minimum(labels[touch], shifted[touch]).astype(np.int64)
            hi = np.maximum(labels[touch], shifted[touch]).astype(np.int64)
            packed = (lo << 32) | hi
            for value in np.unique(packed).tolist():
                pairs.add((int(value >> 32), int(value & 0xFFFFFFFF)))
    return sorted(pairs)


def _pair_separator_stats(
    gt: np.ndarray,
    near_barrier: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    max_id = int(gt.max())
    lookup = np.full((max_id + 1, max_id + 1), -1, dtype=np.int32)
    for index, (id_a, id_b) in enumerate(pairs):
        lookup[id_a, id_b] = index
        lookup[id_b, id_a] = index
    totals = np.zeros(len(pairs), dtype=np.int32)
    covered_counts = np.zeros(len(pairs), dtype=np.int32)
    gap_map = np.zeros(gt.shape, dtype=np.int32)
    for distance in range(1, 9):
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            step_y = dy * distance
            step_x = dx * distance
            shifted = np.zeros_like(gt)
            src_y0 = max(0, -step_y)
            src_y1 = gt.shape[0] - max(0, step_y)
            src_x0 = max(0, -step_x)
            src_x1 = gt.shape[1] - max(0, step_x)
            dst_y0 = max(0, step_y)
            dst_y1 = gt.shape[0] - max(0, -step_y)
            dst_x0 = max(0, step_x)
            dst_x1 = gt.shape[1] - max(0, -step_x)
            if src_y1 <= src_y0 or src_x1 <= src_x0:
                continue
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = gt[src_y0:src_y1, src_x0:src_x1]
        touch = (gt > 0) & (shifted > 0) & (gt != shifted)
        if not np.any(touch):
            continue
        pair_i = lookup[gt[touch], shifted[touch]]
        valid = pair_i >= 0
        if not np.any(valid):
            continue
        np.add.at(totals, pair_i[valid], 1)
        covered_touch = near_barrier[touch]
        np.add.at(covered_counts, pair_i[valid & covered_touch], 1)
        uncovered = touch.copy()
        uncovered[touch] = valid & ~covered_touch
        gap_map[uncovered] = pair_i[valid & ~covered_touch] + 1
    coverage = np.divide(covered_counts, np.maximum(totals, 1)).astype(np.float64)
    gaps = np.zeros(len(pairs), dtype=np.float64)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (gap_map > 0).astype(np.uint8),
        connectivity=8,
    )
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        y = int(stats[component, cv2.CC_STAT_TOP])
        x = int(stats[component, cv2.CC_STAT_LEFT])
        pair_i = int(gap_map[y, x]) - 1
        if pair_i < 0:
            continue
        if 0 <= pair_i < gaps.size:
            gaps[pair_i] = max(gaps[pair_i], float(area))
    return coverage, gaps


def _label_name(code: int) -> str:
    if code == LABEL_METAL:
        return "METAL"
    if code == LABEL_BACKGROUND:
        return "BACKGROUND"
    return "UNKNOWN"


def save_debug(frame_id: str, image: np.ndarray, gt_full: np.ndarray, result, metrics: dict[str, object]) -> None:
    folder = OUTPUT_ROOT / frame_id
    folder.mkdir(parents=True, exist_ok=True)
    gray = crop_evaluation_region(image, CROP_PX, frame_id=frame_id)
    gt = relabel_connected_components(crop_evaluation_region(gt_full, CROP_PX, frame_id=frame_id))
    barrier = crop_evaluation_region(result.barrier_network.barrier, CROP_PX, frame_id=frame_id)
    for name, array in result.debug_images.items():
        cropped = crop_evaluation_region(array, CROP_PX, frame_id=frame_id)
        cv2.imwrite(str(folder / f"{name}.png"), cropped)

    gt_boundary = _boundary(gt)
    missed = gt_boundary & ~(cv2.dilate((barrier > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    interior = cv2.erode((gt > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    false_internal = (barrier > 0) & interior & ~(cv2.dilate(gt_boundary.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[gt_boundary] = (40, 180, 40)
    overlay[barrier > 0] = (40, 40, 220)
    overlay[missed] = (0, 220, 255)
    cv2.imwrite(str(folder / "metal_boundary_gt_overlay.png"), overlay)

    false_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    false_vis[false_internal] = (0, 0, 255)
    cv2.imwrite(str(folder / "metal_boundary_false_barriers.png"), false_vis)

    missed_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    missed_vis[missed] = (0, 255, 255)
    cv2.imwrite(str(folder / "metal_boundary_missed_boundaries.png"), missed_vis)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _printable(metrics: dict[str, object]) -> dict[str, object]:
    copy = dict(metrics)
    purity = dict(copy.get("region_purity", {}))  # type: ignore[arg-type]
    purity.pop("multi_gt_region_ids_sample", None)
    copy["region_purity"] = purity
    plate = copy.get("wide_plate")
    if isinstance(plate, dict) and "regions" in plate:
        plate = dict(plate)
        plate["regions"] = plate["regions"][:5]
        copy["wide_plate"] = plate
    return copy


if __name__ == "__main__":
    raise SystemExit(main())
