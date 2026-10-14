"""benchmark_metal_segmentation: evaluation record."""
from __future__ import annotations
from dataclasses import asdict

def _evaluation_record(
    metrics: SegmentationMetrics,
    *,
    original_shape: tuple[int, int],
    evaluation_shape: tuple[int, int],
    full_metrics: SegmentationMetrics,
    removed_gt_component_count: int,
    removed_predicted_component_count: int,
) -> dict[str, float | int | bool]:
    original_height, original_width = original_shape
    evaluation_height, evaluation_width = evaluation_shape
    return {
        **asdict(metrics),
        "original_height": int(original_height),
        "original_width": int(original_width),
        "evaluation_height": int(evaluation_height),
        "evaluation_width": int(evaluation_width),
        "expected_components_full_frame": full_metrics.expected_components,
        "predicted_components_full_frame": full_metrics.predicted_components,
        "removed_gt_component_count": int(removed_gt_component_count),
        "removed_predicted_component_count": int(removed_predicted_component_count),
        "iou_full_frame": full_metrics.iou,
        "false_positive_components_full_frame": full_metrics.false_positive_components,
        "missed_expected_components_full_frame": full_metrics.missed_expected_components,
        "false_merges_full_frame": full_metrics.false_merges,
        "false_splits_full_frame": full_metrics.false_splits,
        "component_count_absolute_error_full_frame": full_metrics.component_count_absolute_error,
        "topology_exact_match_full_frame": full_metrics.topology_exact_match,
    }
