"""Measure one-parameter sensitivity of the classical metal strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark_metal_segmentation import (
    EVALUATION_BORDER_CROP_PX,
    REAL_DATASET_ROOT,
    _aggregate_strategy_metrics,
    build_real_benchmark_cases,
    run_benchmark,
)

from contour.vision.metal_recovery.strategy_registry import normalize_strategy_parameters

HARD_FRAMES = ("0175", "0580", "3242")
SENSITIVITY_SWEEPS: dict[str, dict[str, tuple[object, ...]]] = {
    "owt_ucm": {
        "hierarchy_level": (0.08, 0.2, 0.45),
        "minimum_contour_strength": (0.06, 0.12, 0.24),
        "watershed_minima_suppression": (0.02, 0.06, 0.15),
    },
    "graph_multi_separator": {
        "separator_projection_min_core_evidence": (0.15, 0.25, 0.4),
        "separator_projection_core_margin": (0.0, 0.1, 0.25),
        "metal_merge_max_separator_confidence": (0.7, 0.85, 0.95),
    },
    "gasp": {
        "minimum_merge_affinity": (-0.05, 0.05, 0.2),
        "maximum_repulsive_conflict": (0.25, 0.45, 0.7),
        "linkage_criterion": ("average", "mutex_abs_max", "sum"),
    },
    "mutex_watershed": {
        "attractive_weight_scale": (0.6, 1.0, 1.6),
        "mutex_weight_scale": (0.6, 1.0, 1.6),
        "minimum_mutex_confidence": (0.4, 0.55, 0.7),
    },
    "multicut": {
        "affinity_bias": (0.35, 0.5, 0.65),
        "repulsion_cost_scale": (0.5, 1.0, 2.0),
        "atomic_region_scale": (10, 16, 24),
    },
    "lifted_multicut": {
        "maximum_lifted_distance": (12, 24, 40),
        "lifted_repulsion_weight": (0.35, 0.75, 1.5),
        "lifted_confidence_threshold": (0.45, 0.6, 0.75),
    },
}


def _compact_case_metrics(metrics: dict[str, object]) -> dict[str, object]:
    selected_metrics = (
        "iou",
        "boundary_f1",
        "false_merges",
        "false_splits",
        "component_count_absolute_error",
        "elapsed_ms",
        "stage_timings_ms",
    )
    return {name: metrics[name] for name in selected_metrics if name in metrics}


def run_sensitivity(
    *,
    dataset_root: Path,
    frame_ids: tuple[str, ...],
    strategies: tuple[str, ...],
) -> dict[str, Any]:
    all_cases = build_real_benchmark_cases(dataset_root)
    available = {case.name for case in all_cases}
    missing = sorted(set(frame_ids) - available)
    if missing:
        raise ValueError(f"Unknown frame ids: {', '.join(missing)}")
    cases = tuple(case for case in all_cases if case.name in set(frame_ids))
    started = perf_counter()
    results: dict[str, object] = {}
    metrics_cache: dict[tuple[str, str, tuple[tuple[str, object], ...]], dict[str, object]] = {}
    for strategy in strategies:
        parameter_results: dict[str, object] = {}
        for parameter_name, values in SENSITIVITY_SWEEPS[strategy].items():
            value_results: list[dict[str, object]] = [{"value": value, "cases": {}} for value in values]
            raw_results: list[dict[str, dict[str, dict[str, object]]]] = [{} for _value in values]
            categories = {case.name: case.category for case in cases}
            # Frame-major ordering keeps the native-resolution feature/graph
            # cache hot across solver-only parameter values without changing
            # any measured output or production cache limits.
            for case in cases:
                for index, value in enumerate(values):
                    overrides = {parameter_name: value}
                    effective = normalize_strategy_parameters(strategy, overrides)
                    cache_key = (case.name, strategy, tuple(effective.items()))
                    metrics = metrics_cache.get(cache_key)
                    if metrics is None:
                        report = run_benchmark(
                            (strategy,),
                            cases=(case,),
                            suite="real_sensitivity",
                            evaluation_stage="ui",
                            evaluation_border_crop_px=EVALUATION_BORDER_CROP_PX,
                            strategy_parameter_overrides={strategy: overrides},
                        )
                        report_cases = report["cases"]
                        assert isinstance(report_cases, dict)
                        metrics = report_cases[case.name][strategy]
                        metrics_cache[cache_key] = metrics
                    raw_results[index][case.name] = {strategy: metrics}
                    compact_cases = value_results[index]["cases"]
                    assert isinstance(compact_cases, dict)
                    compact_cases[case.name] = _compact_case_metrics(metrics)
            for index, _value in enumerate(values):
                value_results[index]["aggregate"] = _aggregate_strategy_metrics(
                    raw_results[index],
                    categories,
                    (strategy,),
                )[strategy]
            parameter_results[parameter_name] = value_results
        results[strategy] = parameter_results
    return {
        "schema_version": 1,
        "suite": "real_sensitivity",
        "frames": list(frame_ids),
        "evaluation_border_crop_px": EVALUATION_BORDER_CROP_PX,
        "elapsed_seconds": perf_counter() - started,
        "sweeps": {
            strategy: {parameter: list(values) for parameter, values in SENSITIVITY_SWEEPS[strategy].items()}
            for strategy in strategies
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REAL_DATASET_ROOT)
    parser.add_argument("--frames", nargs="+", default=list(HARD_FRAMES))
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=tuple(SENSITIVITY_SWEEPS),
        default=list(SENSITIVITY_SWEEPS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "metal_new_strategies_sensitivity.json",
    )
    args = parser.parse_args()
    report = run_sensitivity(
        dataset_root=args.dataset_root,
        frame_ids=tuple(args.frames),
        strategies=tuple(args.strategies),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({report['elapsed_seconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
