"""Search explicit per-frame metal configs for exact topology after crop=50.

Auto is used only as a diagnostic reference. Saved configs are always an
explicit strategy plus a parameter snapshot. Recognition still runs on the
full SEM frame; crop is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from contour.vision.metal_recovery import MetalRecoveryConfig, detect_metalization  # noqa: E402
from contour.vision.metal_recovery.gradient_watershed import GradientWatershedConfig  # noqa: E402
from scripts.benchmark_metal_segmentation import (  # noqa: E402
    REAL_DATASET_ROOT,
    _rasterize_detected_polygons,
    _ui_recovery_config,
    build_real_benchmark_cases,
    measure_segmentation,
    prepare_evaluation_masks,
)

CROP_PX = 50
EXPLICIT_STRATEGIES = (
    "legacy_otsu",
    "gradient_watershed",
    "local_adaptive",
    "reconstruction",
    "closed_boundary",
)
PROBLEM_ORDER = ("2497", "0175", "0580", "1514", "3312", "5101", "3242", "0401")
CHECKPOINT_PATH = PLUGIN_ROOT / "benchmarks" / "perfect_search_checkpoint.json"
PERFECT_PATH = PLUGIN_ROOT / "benchmarks" / "metal_per_frame_perfect_configs.json"
SNAPSHOT_SKIP = {
    "preset_name",
    "contrast_bias",
    "noise_suppression",
}

COARSE_SWEEPS: dict[str, tuple[Any, ...]] = {
    "min_contrast": (40.0, 50.0, 60.0, 70.0, 80.0),
    "min_object_source_contrast": (8.0, 12.0, 16.0, 20.0, 24.0, 28.0),
    "min_object_rim_contrast": (24.0, 36.0, 48.0, 60.0),
    "speckle_removal_px": (0, 1, 2, 3, 4, 6, 8),
    "gap_bridge_px": (0, 1, 2, 3, 4),
    "min_area": (50.0, 95.0, 150.0, 250.0, 400.0),
    "min_component_area": (20.0, 30.0, 60.0, 95.0, 150.0),
    "min_width_px": (3.0, 4.0, 6.0, 8.0, 12.0),
    "min_length_px": (6.0, 8.0, 12.0, 16.0),
    "min_perimeter": (8.0, 10.0, 20.0, 32.0),
    "epsilon_simplify": (0.5, 1.0, 1.5, 2.0),
    "watershed_smoothing_sigma": (0.6, 1.0, 1.4, 2.0),
    "watershed_core_margin": (4.0, 8.0, 12.0, 16.0),
    "watershed_groove_margin": (8.0, 16.0, 24.0),
    "watershed_seed_speckle_px": (2, 4, 6, 8),
    "watershed_valley_span_px": (3, 5, 7, 9),
    "watershed_valley_depth": (30.0, 45.0, 60.0, 80.0),
    "watershed_rim_probe_px": (4, 6, 8),
    "boundary_relief": (8.0, 16.0, 24.0, 32.0),
    "auto_directional_gap_bridge_px": (0, 3, 5),
}


def _empty_state() -> dict[str, Any]:
    return {
        "crop_px": CROP_PX,
        "perfect": {},
        "best": {},
        "library": [],
        "trials": 0,
    }


def load_state() -> dict[str, Any]:
    if CHECKPOINT_PATH.is_file():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return _empty_state()


def save_state(state: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def snapshot_config(config: MetalRecoveryConfig) -> dict[str, Any]:
    payload = config.to_snapshot()
    for key in SNAPSHOT_SKIP:
        payload.pop(key, None)
    return payload


def config_key(strategy: str, parameters: dict[str, Any]) -> str:
    payload = {"strategy": strategy, **parameters}
    return json.dumps(payload, sort_keys=True, default=str)


def topology_error(metrics: dict[str, Any]) -> int:
    return int(
        metrics["false_components"]
        + metrics["misses"]
        + metrics["merges"]
        + metrics["splits"]
    )


def metrics_payload(result) -> dict[str, Any]:
    return {
        "iou": result.iou,
        "precision": result.precision,
        "recall": result.recall,
        "boundary_f1": result.boundary_f1,
        "false_components": result.false_positive_components,
        "misses": result.missed_expected_components,
        "merges": result.false_merges,
        "splits": result.false_splits,
        "count_error": result.component_count_absolute_error,
        "predicted_component_count": result.predicted_components,
        "gt_component_count": result.expected_components,
        "exact_topology": result.topology_exact_match,
    }


def build_config(strategy: str, overrides: dict[str, Any] | None = None) -> MetalRecoveryConfig:
    config = _ui_recovery_config(strategy, GradientWatershedConfig())
    if overrides:
        config = replace(config, **overrides)
    if strategy == "gradient_watershed":
        config = replace(config, use_wide_conductor_gradient=True)
    else:
        config = replace(config, use_wide_conductor_gradient=False)
    return replace(config, segmentation_strategy=strategy)


def evaluate_case(case, config: MetalRecoveryConfig) -> dict[str, Any]:
    detection = detect_metalization(case.image, config, source_image=case.source_image)
    predicted_labels = _rasterize_detected_polygons(detection.accepted, case.image.shape[:2])
    predicted = np.where(predicted_labels > 0, 255, 0).astype(np.uint8)
    predicted_eval, expected_eval, predicted_labels_eval = prepare_evaluation_masks(
        predicted,
        case.labels,
        crop_px=CROP_PX,
        predicted_labels=predicted_labels,
        frame_id=case.name,
    )
    measured = measure_segmentation(
        predicted_eval,
        expected_eval,
        elapsed_ms=0.0,
        predicted_labels=predicted_labels_eval,
    )
    auto_strategy = detection.params_snapshot.get("auto_selected_strategy")
    payload = metrics_payload(measured)
    payload["auto_selected_strategy"] = auto_strategy
    return payload


def record_trial(
    state: dict[str, Any],
    *,
    frame: str,
    strategy: str,
    config: MetalRecoveryConfig,
    metrics: dict[str, Any],
) -> None:
    parameters = snapshot_config(config)
    error = topology_error(metrics)
    trial = {
        "frame": frame,
        "strategy": strategy,
        "parameters": parameters,
        "metrics": metrics,
        "topology_error": error,
    }
    state["trials"] = int(state.get("trials", 0)) + 1
    best = state.setdefault("best", {})
    previous = best.get(frame)
    better = previous is None
    if previous is not None:
        prev_error = int(previous["topology_error"])
        if error < prev_error:
            better = True
        elif error == prev_error == 0:
            if metrics["iou"] > previous["metrics"]["iou"] + 1e-12 or (abs(metrics["iou"] - previous["metrics"]["iou"]) <= 1e-12 and metrics["boundary_f1"] > previous["metrics"]["boundary_f1"]):
                better = True
        elif error == prev_error and error > 0 and metrics["iou"] > previous["metrics"]["iou"] + 1e-12:
            better = True
    if better:
        best[frame] = trial
        print(
            f"  [{frame}] {strategy} err={error} "
            f"false={metrics['false_components']} miss={metrics['misses']} "
            f"merge={metrics['merges']} split={metrics['splits']} "
            f"iou={metrics['iou']:.3f} exact={metrics['exact_topology']}",
            flush=True,
        )
    if metrics["exact_topology"]:
        _assign_perfect(state, frame, strategy, parameters, metrics)
    save_state(state)


def _assign_perfect(
    state: dict[str, Any],
    frame: str,
    strategy: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    key = config_key(strategy, parameters)
    library = state.setdefault("library", [])
    config_id = None
    for item in library:
        if config_key(item["strategy"], item["parameters"]) == key:
            config_id = item["id"]
            if frame not in item["frames"]:
                item["frames"].append(frame)
            break
    if config_id is None:
        config_id = f"C{len(library) + 1:02d}"
        library.append(
            {
                "id": config_id,
                "strategy": strategy,
                "parameters": parameters,
                "frames": [frame],
            }
        )
    current = state.setdefault("perfect", {}).get(frame)
    candidate = {
        "frame": frame,
        "config_id": config_id,
        "strategy": strategy,
        "parameters": parameters,
        "metrics": metrics,
        "topology_error": 0,
    }
    if current is None:
        state["perfect"][frame] = candidate
        return
    if metrics["iou"] > current["metrics"]["iou"] + 1e-12 or (abs(metrics["iou"] - current["metrics"]["iou"]) <= 1e-12 and metrics["boundary_f1"] > current["metrics"]["boundary_f1"]):
        state["perfect"][frame] = candidate


def write_perfect_file(state: dict[str, Any]) -> None:
    frames = sorted(state.get("perfect", {}))
    document = {
        "evaluation_border_crop_px": CROP_PX,
        "exact_topology_frames": len(frames),
        "library": state.get("library", []),
        "frames": [state["perfect"][name] for name in frames],
    }
    PERFECT_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def unresolved_frames(state: dict[str, Any], all_names: list[str]) -> list[str]:
    perfect = set(state.get("perfect", {}))
    ordered = [name for name in PROBLEM_ORDER if name in all_names and name not in perfect]
    leftover = [name for name in all_names if name not in perfect and name not in ordered]
    return ordered + leftover


def probe_explicit_strategies(cases: dict[str, Any], state: dict[str, Any], names: list[str]) -> None:
    for strategy in EXPLICIT_STRATEGIES:
        config = build_config(strategy)
        print(f"== probe {strategy} ==", flush=True)
        for name in names:
            started = perf_counter()
            metrics = evaluate_case(cases[name], config)
            print(f"  {name} {perf_counter() - started:.1f}s", flush=True)
            record_trial(state, frame=name, strategy=strategy, config=config, metrics=metrics)


def try_library_on_frame(cases: dict[str, Any], state: dict[str, Any], frame: str) -> bool:
    for item in list(state.get("library", [])):
        if frame in item.get("frames", []):
            continue
        config = build_config(item["strategy"], _overrides_from_snapshot(item["strategy"], item["parameters"]))
        metrics = evaluate_case(cases[frame], config)
        record_trial(
            state,
            frame=frame,
            strategy=item["strategy"],
            config=config,
            metrics=metrics,
        )
        if metrics["exact_topology"]:
            return True
    return False


def _overrides_from_snapshot(strategy: str, parameters: dict[str, Any]) -> dict[str, Any]:
    baseline = snapshot_config(build_config(strategy))
    return {key: value for key, value in parameters.items() if baseline.get(key) != value}


def sweep_frame(cases: dict[str, Any], state: dict[str, Any], frame: str) -> None:
    best = state.get("best", {}).get(frame)
    strategies = [best["strategy"]] if best else list(EXPLICIT_STRATEGIES)
    if best and best["strategy"] not in EXPLICIT_STRATEGIES:
        strategies = list(EXPLICIT_STRATEGIES)
    seen: set[str] = set()
    for strategy in strategies:
        base = build_config(strategy)
        fields = list(COARSE_SWEEPS)
        if strategy not in {"gradient_watershed", "random_walker", "graph_cut", "reconstruction", "closed_boundary"}:
            fields = [name for name in fields if not name.startswith("watershed") and name != "boundary_relief"]
        for field_name in fields:
            if frame in state.get("perfect", {}):
                return
            for value in COARSE_SWEEPS[field_name]:
                config = replace(base, **{field_name: value})
                key = config_key(strategy, snapshot_config(config))
                if key in seen:
                    continue
                seen.add(key)
                metrics = evaluate_case(cases[frame], config)
                record_trial(state, frame=frame, strategy=strategy, config=config, metrics=metrics)
                if metrics["exact_topology"]:
                    return
        if frame in state.get("perfect", {}):
            return
        refined = state.get("best", {}).get(frame)
        if refined is None or refined["strategy"] != strategy:
            continue
        overrides = _overrides_from_snapshot(strategy, refined["parameters"])
        for field_name, values in COARSE_SWEEPS.items():
            if field_name in overrides:
                continue
            if frame in state.get("perfect", {}):
                return
            for value in values:
                trial_overrides = {**overrides, field_name: value}
                config = build_config(strategy, trial_overrides)
                key = config_key(strategy, snapshot_config(config))
                if key in seen:
                    continue
                seen.add(key)
                metrics = evaluate_case(cases[frame], config)
                record_trial(state, frame=frame, strategy=strategy, config=config, metrics=metrics)
                if metrics["exact_topology"]:
                    return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe", "search", "all"), default="all")
    parser.add_argument("--frames", nargs="*", default=None)
    args = parser.parse_args()
    cases = {case.name: case for case in build_real_benchmark_cases(REAL_DATASET_ROOT)}
    names = args.frames or list(cases)
    state = load_state()
    if args.phase in {"probe", "all"}:
        probe_explicit_strategies(cases, state, names)
        write_perfect_file(state)
    if args.phase in {"search", "all"}:
        for frame in unresolved_frames(state, names):
            print(f"== search {frame} ==", flush=True)
            if try_library_on_frame(cases, state, frame):
                write_perfect_file(state)
                continue
            sweep_frame(cases, state, frame)
            write_perfect_file(state)
            if frame not in state.get("perfect", {}):
                best = state.get("best", {}).get(frame, {})
                print(f"UNRESOLVED {frame}: {json.dumps(best.get('metrics', {}), indent=2)}", flush=True)
    write_perfect_file(state)
    perfect = state.get("perfect", {})
    print(f"exact {len(perfect)}/{len(names)}", flush=True)
    return 0 if len(perfect) == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
