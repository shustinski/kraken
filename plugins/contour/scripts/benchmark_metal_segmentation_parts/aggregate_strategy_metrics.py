"""benchmark_metal_segmentation: aggregate strategy metrics."""
from __future__ import annotations
import numpy as np

def _aggregate_strategy_metrics(
    case_results: dict[str, dict[str, dict[str, object]]],
    case_categories: dict[str, str],
    strategy_names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    aggregates: dict[str, dict[str, object]] = {}
    for strategy in strategy_names:
        metrics = [case_results[name][strategy] for name in case_results]
        positive_metrics = [case_results[name][strategy] for name in case_results if case_categories[name] != "empty"]
        empty_metrics = [case_results[name][strategy] for name in case_results if case_categories[name] == "empty"]
        stage_names = sorted(
            {
                stage_name
                for item in metrics
                for stage_name in (
                    item.get("stage_timings_ms", {}) if isinstance(item.get("stage_timings_ms"), dict) else {}
                )
            }
        )
        aggregates[strategy] = {
            "mean_iou": float(np.mean([item["iou"] for item in metrics])),
            "positive_mean_iou": float(np.mean([item["iou"] for item in positive_metrics])),
            "positive_median_iou": float(np.median([item["iou"] for item in positive_metrics])),
            "mean_precision": float(np.mean([item["precision"] for item in metrics])),
            "positive_mean_precision": float(np.mean([item["precision"] for item in positive_metrics])),
            "mean_recall": float(np.mean([item["recall"] for item in metrics])),
            "positive_mean_recall": float(np.mean([item["recall"] for item in positive_metrics])),
            "mean_boundary_f1": float(np.mean([item["boundary_f1"] for item in metrics])),
            "positive_mean_boundary_f1": float(np.mean([item["boundary_f1"] for item in positive_metrics])),
            "false_merges": int(sum(item["false_merges"] for item in metrics)),
            "false_splits": int(sum(item["false_splits"] for item in metrics)),
            "false_positive_components": int(sum(item["false_positive_components"] for item in metrics)),
            "missed_expected_components": int(sum(item["missed_expected_components"] for item in metrics)),
            "component_count_absolute_error": int(sum(item["component_count_absolute_error"] for item in metrics)),
            "positive_mean_component_precision": float(
                np.mean([item["component_precision"] for item in positive_metrics])
            ),
            "positive_mean_component_recall": float(np.mean([item["component_recall"] for item in positive_metrics])),
            "positive_mean_component_f1": float(np.mean([item["component_f1"] for item in positive_metrics])),
            "empty_false_metal_fraction": float(
                np.mean([item["false_metal_fraction"] for item in empty_metrics]) if empty_metrics else 0.0
            ),
            "exact_topology_frames": int(sum(bool(item["topology_exact_match"]) for item in metrics)),
            "positive_exact_topology_frames": int(sum(bool(item["topology_exact_match"]) for item in positive_metrics)),
            "elapsed_ms": float(sum(item["elapsed_ms"] for item in metrics)),
            "mean_elapsed_ms": float(np.mean([item["elapsed_ms"] for item in metrics])),
            "mean_segmentation_ms": float(np.mean([item["segmentation_ms"] for item in metrics])),
            "mean_refinement_ms": float(np.mean([item["refinement_ms"] for item in metrics])),
            "mean_stage_timings_ms": {
                stage_name: float(
                    np.mean([float(item.get("stage_timings_ms", {}).get(stage_name, 0.0)) for item in metrics])
                )
                for stage_name in stage_names
            },
        }
    return aggregates
