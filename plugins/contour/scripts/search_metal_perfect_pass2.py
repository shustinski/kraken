"""Second-pass local search around best remaining metal configs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
for extra in (SOURCE_ROOT, PLUGIN_ROOT):
    text = str(extra)
    if text not in sys.path:
        sys.path.insert(0, text)

from contour.vision.metal_recovery import detect_metalization  # noqa: E402
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    REAL_DATASET_ROOT,
    _rasterize_detected_polygons,
    build_real_benchmark_cases,
    prepare_evaluation_masks,
)
from scripts.search_metal_perfect_configs import (  # noqa: E402
    build_config,
    evaluate_case,
    load_state,
    record_trial,
    snapshot_config,
    write_perfect_file,
)

CROP = 50
REMAINING = ("2497", "0175", "0580", "1514", "3312", "5101", "3242")
ALT_STRATEGIES = {
    "2497": ("gradient_watershed", "legacy_otsu"),
    "0175": ("local_adaptive", "legacy_otsu", "reconstruction"),
    "0580": ("local_adaptive", "legacy_otsu"),
    "1514": ("local_adaptive", "gradient_watershed", "legacy_otsu"),
    "3312": ("gradient_watershed", "local_adaptive"),
    "5101": ("gradient_watershed", "legacy_otsu"),
    "3242": ("gradient_watershed", "legacy_otsu"),
}


def _overrides(strategy: str, parameters: dict) -> dict:
    baseline = snapshot_config(build_config(strategy))
    return {key: value for key, value in parameters.items() if baseline.get(key) != value}


def localize_failures(case, config) -> dict:
    detection = detect_metalization(case.image, config, source_image=case.source_image)
    pred_full = _rasterize_detected_polygons(detection.accepted, case.image.shape[:2])
    _pred_eval, gt_eval, pred_labels = prepare_evaluation_masks(
        np.where(pred_full > 0, 255, 0).astype(np.uint8),
        case.labels,
        crop_px=CROP,
        predicted_labels=pred_full,
        frame_id=case.name,
    )
    assert pred_labels is not None
    pred_ids = np.unique(pred_labels[pred_labels > 0])
    gt_ids = np.unique(gt_eval[gt_eval > 0])
    expected_areas = np.bincount(gt_eval.ravel())
    predicted_areas = np.bincount(pred_labels.ravel())
    false_ids = []
    for pred_id in pred_ids:
        overlap = gt_eval[pred_labels == pred_id]
        if np.count_nonzero(overlap > 0) / max(1, int(overlap.size)) < 0.10:
            ys, xs = np.nonzero(pred_labels == pred_id)
            false_ids.append(
                {
                    "id": int(pred_id),
                    "area": int(ys.size),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                }
            )
    misses = []
    for gt_id in gt_ids:
        overlap = pred_labels[gt_eval == gt_id]
        if np.count_nonzero(overlap > 0) / max(1, int(overlap.size)) < 0.50:
            ys, xs = np.nonzero(gt_eval == gt_id)
            misses.append(
                {
                    "id": int(gt_id),
                    "area": int(ys.size),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                }
            )
    splits = []
    for gt_id in gt_ids:
        overlap = pred_labels[gt_eval == gt_id]
        expected_area = max(1, int(overlap.size))
        material = 0
        for pred_id in np.unique(overlap[overlap > 0]):
            intersection = int(np.count_nonzero(overlap == pred_id))
            prediction_area = max(1, int(predicted_areas[int(pred_id)]))
            if intersection / expected_area >= 0.01 and intersection / prediction_area >= 0.10:
                material += 1
        if material > 1:
            splits.append({"id": int(gt_id), "area": expected_area, "parts": material})
    merges = []
    for pred_id in pred_ids:
        overlap = gt_eval[pred_labels == pred_id]
        covered = 0
        for gt_id in np.unique(overlap[overlap > 0]):
            intersection = int(np.count_nonzero(overlap == gt_id))
            expected_area = max(1, int(expected_areas[int(gt_id)]))
            if intersection / expected_area >= 0.10:
                covered += 1
        if covered > 1:
            ys, xs = np.nonzero(pred_labels == pred_id)
            merges.append({"id": int(pred_id), "area": int(ys.size), "gt_objects": covered})
    return {"false": false_ids, "misses": misses, "splits": splits, "merges": merges}


def neighborhoods(frame: str, strategy: str, current: dict) -> list[dict]:
    targeted: list[dict] = [dict(current)]
    extras = {
        "2497": [
            {**current, "min_contrast": 50.0, "min_object_source_contrast": 12.0, "min_area": 35.0, "min_width_px": 3.0},
            {**current, "min_contrast": 55.0, "min_object_source_contrast": 14.0, "min_area": 50.0},
            {**current, "min_contrast": 60.0, "min_object_source_contrast": 12.0, "min_area": 50.0, "min_width_px": 3.0},
            {**current, "min_contrast": 60.0, "min_object_source_contrast": 16.0, "min_area": 35.0},
            {**current, "min_contrast": 60.0, "min_object_source_contrast": 16.0, "min_width_px": 3.0, "min_length_px": 6.0},
            {**current, "min_contrast": 50.0, "min_object_source_contrast": 16.0, "min_area": 50.0},
        ],
        "0580": [
            {**current, "watershed_smoothing_sigma": 0.7, "epsilon_simplify": 0.5, "gap_bridge_px": 1},
            {**current, "watershed_smoothing_sigma": 0.6, "speckle_removal_px": 4, "epsilon_simplify": 0.5},
            {**current, "watershed_smoothing_sigma": 0.8, "epsilon_simplify": 0.5},
            {**current, "watershed_smoothing_sigma": 0.6, "gap_bridge_px": 1, "epsilon_simplify": 1.0},
            {**current, "watershed_smoothing_sigma": 0.6, "epsilon_simplify": 0.8, "gap_bridge_px": 2},
        ],
        "0175": [
            {**current, "watershed_smoothing_sigma": 1.4, "watershed_valley_depth": 60.0, "epsilon_simplify": 0.5},
            {**current, "watershed_smoothing_sigma": 2.0, "watershed_valley_span_px": 3, "watershed_valley_depth": 70.0},
            {**current, "watershed_smoothing_sigma": 1.0, "watershed_groove_margin": 8.0, "watershed_core_margin": 4.0},
            {**current, "gap_bridge_px": 0, "speckle_removal_px": 1, "watershed_smoothing_sigma": 1.6},
            {**current, "watershed_smoothing_sigma": 2.0, "watershed_valley_depth": 80.0, "watershed_valley_span_px": 2},
        ],
        "1514": [
            {**current, "min_area": 150.0, "min_width_px": 6.0, "speckle_removal_px": 0},
            {**current, "min_object_source_contrast": 20.0, "min_area": 120.0},
            {**current, "min_contrast": 60.0, "min_area": 150.0, "speckle_removal_px": 2},
            {**current, "min_area": 250.0, "min_width_px": 8.0, "speckle_removal_px": 1},
            {**current, "min_object_source_contrast": 24.0, "min_width_px": 6.0, "min_area": 95.0},
        ],
        "5101": [
            {**current, "min_contrast": 80.0, "min_area": 150.0, "speckle_removal_px": 2},
            {**current, "min_contrast": 80.0, "min_width_px": 6.0, "min_area": 120.0},
            {**current, "min_contrast": 70.0, "min_area": 180.0, "speckle_removal_px": 3},
            {**current, "min_contrast": 80.0, "min_area": 250.0, "min_object_source_contrast": 16.0},
            {**current, "min_contrast": 90.0, "min_area": 150.0, "min_width_px": 6.0},
        ],
        "3312": [
            {**current, "gap_bridge_px": 0, "speckle_removal_px": 1, "epsilon_simplify": 0.5},
            {**current, "min_contrast": 60.0, "epsilon_simplify": 0.5},
            {**current, "epsilon_simplify": 0.5, "min_object_source_contrast": 16.0},
        ],
        "3242": [
            {**current, "min_contrast": 70.0, "speckle_removal_px": 1, "gap_bridge_px": 0, "min_object_source_contrast": 16.0},
            {**current, "min_contrast": 80.0, "speckle_removal_px": 0, "min_width_px": 3.0},
            {**current, "min_contrast": 70.0, "speckle_removal_px": 1, "min_object_source_contrast": 20.0, "gap_bridge_px": 0},
        ],
    }
    targeted = targeted + extras.get(frame, [])
    grid: dict[str, tuple] = {
        "min_contrast": (40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 90.0),
        "min_object_source_contrast": (4.0, 8.0, 12.0, 16.0, 20.0, 28.0),
        "speckle_removal_px": (0, 1, 2, 3, 4, 6),
        "gap_bridge_px": (0, 1, 2, 3),
        "min_area": (20.0, 35.0, 50.0, 95.0, 150.0, 250.0),
        "min_width_px": (2.0, 3.0, 4.0, 6.0, 8.0),
        "epsilon_simplify": (0.5, 1.0, 1.5),
    }
    if strategy in {"gradient_watershed", "reconstruction", "closed_boundary"}:
        grid.update(
            {
                "watershed_smoothing_sigma": (0.5, 0.6, 0.8, 1.0, 1.6, 2.0),
                "watershed_valley_depth": (30.0, 45.0, 60.0, 80.0),
                "watershed_valley_span_px": (2, 3, 5, 7),
            }
        )
    variants = list(targeted)
    for field, values in grid.items():
        for value in values:
            variant = dict(current)
            variant[field] = value
            variants.append(variant)
    unique: list[dict] = []
    seen: set[str] = set()
    for variant in variants:
        key = json.dumps(variant, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def main() -> int:
    cases = {case.name: case for case in build_real_benchmark_cases(REAL_DATASET_ROOT)}
    state = load_state()
    report: dict = {}
    for frame in REMAINING:
        if frame in state.get("perfect", {}):
            continue
        best = state["best"][frame]
        strategy = best["strategy"]
        config = build_config(strategy, _overrides(strategy, best["parameters"]))
        report[frame] = {"baseline": best["metrics"], "failures": localize_failures(cases[frame], config)}
        print(frame, "failures", json.dumps(report[frame]["failures"], default=str)[:900], flush=True)
        current = _overrides(strategy, best["parameters"])
        for candidate_strategy in (strategy, *ALT_STRATEGIES[frame]):
            if frame in state.get("perfect", {}):
                break
            seed = current if candidate_strategy == strategy else {}
            variants = neighborhoods(frame, candidate_strategy, seed)
            print(f"== {frame} {candidate_strategy} variants={len(variants)} ==", flush=True)
            for variant in variants:
                if frame in state.get("perfect", {}):
                    break
                trial_config = build_config(candidate_strategy, variant)
                metrics = evaluate_case(cases[frame], trial_config)
                record_trial(
                    state,
                    frame=frame,
                    strategy=candidate_strategy,
                    config=trial_config,
                    metrics=metrics,
                )
                if metrics["exact_topology"]:
                    print(f"SOLVED {frame} with {candidate_strategy}", flush=True)
                    break
        write_perfect_file(state)
        if frame not in state.get("perfect", {}):
            latest = state["best"][frame]
            print("still", frame, latest["topology_error"], latest["metrics"], flush=True)
    (PLUGIN_ROOT / "benchmarks" / "remaining_failure_localization.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    write_perfect_file(state)
    print("exact", len(state.get("perfect", {})), "/ 23", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
