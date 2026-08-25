"""Ablation benchmark for structural marker-controlled watershed.

Runs S0 (current gradient_watershed) and S1–S3 of the new structural strategy
on the real SEM set. Recognition is always full-frame; metrics use crop=50.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

from contour.vision.metal_recovery.gradient_watershed import (  # noqa: E402
    GradientWatershedConfig,
    gradient_watershed_mask,
)
from contour.vision.metal_recovery.structural_watershed import (  # noqa: E402
    INSTANCE_IDENTITY_VARIANTS,
    clamped_structural_watershed_config,
    run_structural_watershed,
)
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    EVALUATION_BORDER_CROP_PX,
    REAL_DATASET_ROOT,
    build_real_benchmark_cases,
    crop_evaluation_region,
    measure_segmentation,
    relabel_connected_components,
)

HARD_FRAMES = ("0175", "0580", "3242")
EXACT_TOPOLOGY_FRAMES = (
    "0001",
    "0004",
    "0008",
    "0148",
    "0227",
    "0250",
    "0284",
    "0401",
    "0480",
    "0501",
    "0673",
    "0866",
    "1066",
    "1170",
    "4335",
    "4498",
)
DEBUG_KEYS = (
    "metal_structural_denoised",
    "metal_structural_gx",
    "metal_structural_gy",
    "metal_structural_gradient_magnitude",
    "metal_structural_orientation",
    "metal_structural_coherence",
    "metal_structural_ridge_response",
    "metal_structural_ridge_markers_raw",
    "metal_structural_ridge_markers",
    "metal_structural_ridge_fragments",
    "metal_structural_ridge_links_accepted",
    "metal_structural_ridge_links_rejected",
    "metal_structural_ridge_links_boundary_veto",
    "metal_structural_logical_ridge",
    "metal_structural_wide_interior_markers",
    "metal_structural_wide_fragments",
    "metal_structural_logical_wide",
    "metal_structural_foreground_markers",
    "metal_structural_logical_markers",
    "metal_structural_conductor_bands",
    "metal_structural_transverse_samples",
    "metal_structural_band_groups_accepted",
    "metal_structural_band_groups_rejected",
    "metal_structural_background_markers",
    "metal_structural_boundary_cost",
    "metal_structural_watershed_labels",
    "metal_structural_instance_labels",
    "metal_structural_label_boundary",
    "metal_structural_final_mask",
)


def _metrics_dict(metrics) -> dict[str, float | int | bool]:
    payload = asdict(metrics)
    return {
        "iou": float(payload["iou"]),
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "boundary_f1": float(payload["boundary_f1"]),
        "gt_component_count": int(payload["expected_components"]),
        "predicted_component_count": int(payload["predicted_components"]),
        "false_components": int(payload["false_positive_components"]),
        "misses": int(payload["missed_expected_components"]),
        "merges": int(payload["false_merges"]),
        "splits": int(payload["false_splits"]),
        "count_error": int(payload["component_count_absolute_error"]),
        "exact_topology": bool(payload["topology_exact_match"]),
        "runtime_ms": float(payload["elapsed_ms"]),
    }


def remap_positive_ids(labels: np.ndarray) -> np.ndarray:
    """Renumber remaining instance IDs after crop without 8-connected merging."""

    remapped = np.zeros(labels.shape, dtype=np.int32)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return remapped
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    lookup[ids] = np.arange(1, int(ids.size) + 1, dtype=np.int32)
    positive = labels > 0
    remapped[positive] = lookup[labels[positive]]
    return remapped


def _component_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return int(count - 1)


def _percentile_or_zero(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _logical_marker_gt_diagnostics(
    *,
    gt_labels: np.ndarray,
    logical_labels: np.ndarray,
) -> dict[str, int | float]:
    metal = gt_labels > 0
    gt_ids = np.unique(gt_labels[metal])
    marker_ids = np.unique(logical_labels[logical_labels > 0])
    markers_per_gt: dict[int, int] = {int(gt_id): 0 for gt_id in gt_ids}
    overlapping_zero = 0
    overlapping_one = 0
    overlapping_multi = 0
    for marker_id in marker_ids:
        region = logical_labels == marker_id
        covered = np.unique(gt_labels[region])
        covered = covered[covered > 0]
        if covered.size == 0:
            overlapping_zero += 1
        elif covered.size == 1:
            overlapping_one += 1
        else:
            overlapping_multi += 1
        for gt_id in covered:
            markers_per_gt[int(gt_id)] += 1
    per_gt_values = list(markers_per_gt.values()) if markers_per_gt else [0]
    return {
        "gt_component_count": int(gt_ids.size),
        "logical_marker_count": int(marker_ids.size),
        "gt_without_logical_marker": int(sum(1 for count in per_gt_values if count == 0)),
        "gt_with_exactly_one_logical_marker": int(sum(1 for count in per_gt_values if count == 1)),
        "gt_with_more_than_one_logical_marker": int(sum(1 for count in per_gt_values if count > 1)),
        "logical_overlapping_zero_gt": overlapping_zero,
        "logical_overlapping_exactly_one_gt": overlapping_one,
        "logical_overlapping_more_than_one_gt": overlapping_multi,
        "mean_markers_per_gt": float(np.mean(per_gt_values)) if per_gt_values else 0.0,
        "median_markers_per_gt": float(np.median(per_gt_values)) if per_gt_values else 0.0,
        "p90_markers_per_gt": _percentile_or_zero(per_gt_values, 90.0),
    }


def _marker_gt_diagnostics(
    *,
    gt_labels: np.ndarray,
    ridge_raw: np.ndarray,
    ridge_linked: np.ndarray,
    wide_markers: np.ndarray,
    fg_markers: np.ndarray,
    logical_labels: np.ndarray | None = None,
) -> dict[str, int | float | list[int]]:
    metal = gt_labels > 0
    gt_ids = np.unique(gt_labels[metal])
    _fg_n, fg_cc = cv2.connectedComponents((fg_markers > 0).astype(np.uint8), connectivity=8)
    marker_ids = np.unique(fg_cc[fg_cc > 0])
    overlapping_metal = 0
    overlapping_multi_gt = 0
    markers_per_gt: dict[int, int] = {int(gt_id): 0 for gt_id in gt_ids}
    for marker_id in marker_ids:
        region = fg_cc == marker_id
        if np.any(region & metal):
            overlapping_metal += 1
        covered = np.unique(gt_labels[region])
        covered = covered[covered > 0]
        if covered.size > 1:
            overlapping_multi_gt += 1
        for gt_id in covered:
            markers_per_gt[int(gt_id)] += 1
    per_gt_values = list(markers_per_gt.values()) if markers_per_gt else [0]
    gt_without = int(sum(1 for count in per_gt_values if count == 0))
    gt_multi = int(sum(1 for count in per_gt_values if count > 1))
    payload: dict[str, int | float | list[int]] = {
        "gt_component_count": int(gt_ids.size),
        "raw_ridge_cc_count": _component_count(ridge_raw),
        "linked_ridge_marker_count": _component_count(ridge_linked),
        "wide_interior_marker_count": _component_count(wide_markers),
        "combined_unique_foreground_marker_count": int(marker_ids.size),
        "markers_overlapping_actual_metal": overlapping_metal,
        "mean_markers_per_gt_object": float(np.mean(per_gt_values)) if per_gt_values else 0.0,
        "gt_objects_without_any_marker": gt_without,
        "gt_objects_containing_more_than_one_marker": gt_multi,
        "markers_overlapping_more_than_one_gt_object": overlapping_multi_gt,
    }
    if logical_labels is not None and logical_labels.size and logical_labels.shape == gt_labels.shape:
        payload.update(_logical_marker_gt_diagnostics(gt_labels=gt_labels, logical_labels=logical_labels))
    return payload


def _segment_variant(
    image: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray], float, int, int, dict[str, object]]:
    watershed_config = GradientWatershedConfig()
    started = perf_counter()
    if variant == "s0":
        mask = gradient_watershed_mask(image, watershed_config)
        debug: dict[str, np.ndarray] = {"metal_structural_final_mask": mask}
        elapsed_ms = (perf_counter() - started) * 1000.0
        return mask, None, debug, elapsed_ms, 0, 0, {}
    result = run_structural_watershed(
        image,
        watershed_config,
        clamped_structural_watershed_config(variant=variant),
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    return (
        result.mask,
        result.instance_labels,
        result.debug_images,
        elapsed_ms,
        int(result.instance_count),
        int(result.label_fragment_count),
        asdict(result.consolidation),
    )


def _save_debug(directory: Path, frame: str, variant: str, debug: dict[str, np.ndarray]) -> None:
    frame_dir = directory / variant / frame
    frame_dir.mkdir(parents=True, exist_ok=True)
    for key in DEBUG_KEYS:
        image = debug.get(key)
        if image is None:
            continue
        path = frame_dir / f"{key}.png"
        cv2.imwrite(str(path), image)


def run_structural_ablation(
    *,
    dataset_root: Path,
    frames: tuple[str, ...] | None,
    variants: tuple[str, ...],
    crop_px: int,
    diagnostics_dir: Path | None,
) -> dict[str, object]:
    cases = build_real_benchmark_cases(dataset_root)
    selected = [case for case in cases if frames is None or case.name in frames]
    report: dict[str, object] = {
        "evaluation_border_crop_px": crop_px,
        "variants": list(variants),
        "frames": [case.name for case in selected],
        "results": {},
    }
    results: dict[str, dict[str, dict[str, float | int | bool]]] = {}
    for variant in variants:
        results[variant] = {}
        exact = 0
        for case in selected:
            mask, labels, debug, elapsed_ms, instance_count, fragment_count, consolidation = _segment_variant(
                case.image,
                variant,
            )
            predicted = crop_evaluation_region(mask, crop_px, frame_id=case.name)
            expected = crop_evaluation_region(case.labels, crop_px, frame_id=case.name)
            if crop_px > 0:
                expected = relabel_connected_components(expected)
            predicted_labels = None
            identity_source = "binary_cc"
            if variant in INSTANCE_IDENTITY_VARIANTS and labels is not None:
                predicted_labels = remap_positive_ids(
                    crop_evaluation_region(labels, crop_px, frame_id=case.name)
                )
                identity_source = "instance_labels"
            metrics = measure_segmentation(
                predicted,
                expected,
                elapsed_ms=elapsed_ms,
                predicted_labels=predicted_labels,
            )
            payload = _metrics_dict(metrics)
            payload["instance_count"] = instance_count
            payload["label_fragment_count"] = fragment_count
            payload["identity_source"] = identity_source
            payload["consolidation"] = consolidation
            if predicted_labels is not None and np.any(predicted_labels > 0):
                areas = np.bincount(predicted_labels.ravel())
                payload["largest_instance_area"] = int(areas[1:].max()) if areas.size > 1 else 0
            else:
                payload["largest_instance_area"] = 0
            if case.name in HARD_FRAMES and variant != "s0":
                logical_source = debug.get("metal_structural_logical_markers_i32")
                logical_labels = None
                if logical_source is not None:
                    logical_labels = remap_positive_ids(
                        crop_evaluation_region(logical_source, crop_px, frame_id=case.name)
                    )
                payload["marker_diagnostics"] = _marker_gt_diagnostics(
                    gt_labels=expected,
                    ridge_raw=crop_evaluation_region(
                        debug.get("metal_structural_ridge_markers_raw", mask),
                        crop_px,
                        frame_id=case.name,
                    ),
                    ridge_linked=crop_evaluation_region(
                        debug.get("metal_structural_ridge_markers", mask),
                        crop_px,
                        frame_id=case.name,
                    ),
                    wide_markers=crop_evaluation_region(
                        debug.get("metal_structural_wide_interior_markers", mask),
                        crop_px,
                        frame_id=case.name,
                    ),
                    fg_markers=crop_evaluation_region(
                        debug.get("metal_structural_foreground_markers", mask),
                        crop_px,
                        frame_id=case.name,
                    ),
                    logical_labels=logical_labels,
                )
            results[variant][case.name] = payload
            if payload["exact_topology"]:
                exact += 1
            if diagnostics_dir is not None and case.name in HARD_FRAMES and variant != "s0":
                _save_debug(diagnostics_dir, case.name, variant, debug)
            marker_note = ""
            diagnostics = payload.get("marker_diagnostics")
            if isinstance(diagnostics, dict) and "logical_marker_count" in diagnostics:
                marker_note = (
                    f" logical={diagnostics['logical_marker_count']}"
                    f" mean/gt={diagnostics['mean_markers_per_gt']:.2f}"
                    f" median/gt={diagnostics['median_markers_per_gt']:.2f}"
                    f" gt>1={diagnostics['gt_with_more_than_one_logical_marker']}"
                    f" cross={diagnostics['logical_overlapping_more_than_one_gt']}"
                )
            print(
                f"{variant} {case.name}: "
                f"iou={payload['iou']:.3f} "
                f"bf1={payload['boundary_f1']:.3f} "
                f"false={payload['false_components']} "
                f"miss={payload['misses']} "
                f"merge={payload['merges']} "
                f"split={payload['splits']} "
                f"gt={payload['gt_component_count']} "
                f"pred={payload['predicted_component_count']} "
                f"inst={instance_count} "
                f"frag={fragment_count} "
                f"exact={payload['exact_topology']} "
                f"ms={payload['runtime_ms']:.0f}"
                f"{marker_note}",
                flush=True,
            )
        report_variant = {
            "exact_topology_frames": exact,
            "total_merges": int(sum(item["merges"] for item in results[variant].values())),
            "total_misses": int(sum(item["misses"] for item in results[variant].values())),
            "total_splits": int(sum(item["splits"] for item in results[variant].values())),
            "total_false": int(sum(item["false_components"] for item in results[variant].values())),
            "mean_iou": float(np.mean([item["iou"] for item in results[variant].values()])),
            "mean_runtime_ms": float(np.mean([item["runtime_ms"] for item in results[variant].values()])),
        }
        report.setdefault("aggregates", {})[variant] = report_variant  # type: ignore[index]
        print(
            f"{variant} summary: exact={exact}/{len(selected)} "
            f"merges={report_variant['total_merges']} "
            f"misses={report_variant['total_misses']} "
            f"splits={report_variant['total_splits']} "
            f"mean_iou={report_variant['mean_iou']:.3f} "
            f"mean_ms={report_variant['mean_runtime_ms']:.0f}",
            flush=True,
        )
    report["results"] = results
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REAL_DATASET_ROOT)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--variants", nargs="+", default=["s11", "s12", "s13", "s14"])
    parser.add_argument("--evaluation-border-crop-px", type=int, default=EVALUATION_BORDER_CROP_PX)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument(
        "--set",
        choices=("hard", "exact", "all"),
        default="hard",
        help="hard=0175/0580/3242, exact=16 current perfect frames, all=23-frame set",
    )
    args = parser.parse_args()
    frames: tuple[str, ...] | None
    if args.frames:
        frames = tuple(args.frames)
    elif args.set == "hard":
        frames = HARD_FRAMES
    elif args.set == "exact":
        frames = EXACT_TOPOLOGY_FRAMES
    else:
        frames = None
    report = run_structural_ablation(
        dataset_root=args.dataset_root,
        frames=frames,
        variants=tuple(args.variants),
        crop_px=int(args.evaluation_border_crop_px),
        diagnostics_dir=args.diagnostics_dir,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
