"""benchmark_metal_segmentation: measure segmentation."""
from __future__ import annotations
import numpy as np

from .rasterize_polygon_labels import _boundary_f1

from .render_sem import SegmentationMetrics

from .topology_errors import _topology_errors

def measure_segmentation(
    predicted: np.ndarray,
    expected_labels: np.ndarray,
    *,
    elapsed_ms: float,
    segmentation_ms: float | None = None,
    refinement_ms: float = 0.0,
    predicted_labels: np.ndarray | None = None,
) -> SegmentationMetrics:
    predicted_active = predicted > 0
    expected_active = expected_labels > 0
    true_positive = int(np.count_nonzero(predicted_active & expected_active))
    false_positive = int(np.count_nonzero(predicted_active & ~expected_active))
    false_negative = int(np.count_nonzero(~predicted_active & expected_active))
    total_pixels = max(1, int(predicted_active.size))
    expected_pixels = int(np.count_nonzero(expected_active))
    predicted_pixels = int(np.count_nonzero(predicted_active))
    union = true_positive + false_positive + false_negative
    (
        predicted_components,
        false_merges,
        false_splits,
        false_positive_components,
        missed_expected_components,
        matched_expected_components,
    ) = _topology_errors(predicted, expected_labels, predicted_labels)
    supported_components = max(0, predicted_components - false_positive_components)
    expected_components = int(np.unique(expected_labels[expected_labels > 0]).size)
    component_precision = (
        1.0
        if predicted_components == 0 and expected_components == 0
        else supported_components / max(1, predicted_components)
    )
    component_recall = 1.0 if expected_components == 0 else matched_expected_components / expected_components
    component_f1 = (
        0.0
        if component_precision + component_recall <= 0.0
        else 2.0 * component_precision * component_recall / (component_precision + component_recall)
    )
    topology_exact_match = (
        false_positive_components == 0
        and missed_expected_components == 0
        and false_merges == 0
        and false_splits == 0
        and predicted_components == expected_components
    )
    return SegmentationMetrics(
        iou=1.0 if union == 0 else true_positive / union,
        precision=(1.0 if predicted_pixels == 0 and expected_pixels == 0 else true_positive / max(1, predicted_pixels)),
        recall=(1.0 if expected_pixels == 0 and predicted_pixels == 0 else true_positive / max(1, expected_pixels)),
        boundary_f1=_boundary_f1(predicted, expected_active.astype(np.uint8) * 255),
        expected_components=expected_components,
        predicted_components=predicted_components,
        false_merges=false_merges,
        false_splits=false_splits,
        false_positive_components=false_positive_components,
        missed_expected_components=missed_expected_components,
        matched_expected_components=matched_expected_components,
        component_count_absolute_error=abs(predicted_components - expected_components),
        component_precision=component_precision,
        component_recall=component_recall,
        component_f1=component_f1,
        false_merge_indicator=false_merges > 0,
        false_split_indicator=false_splits > 0,
        predicted_foreground_fraction=predicted_pixels / total_pixels,
        gt_foreground_fraction=expected_pixels / total_pixels,
        false_positive_area_fraction=false_positive / total_pixels,
        false_negative_area_fraction=false_negative / total_pixels,
        false_metal_fraction=predicted_pixels / total_pixels if expected_pixels == 0 else 0.0,
        segmentation_ms=float(elapsed_ms if segmentation_ms is None else segmentation_ms),
        refinement_ms=float(refinement_ms),
        elapsed_ms=float(elapsed_ms),
        topology_exact_match=topology_exact_match,
    )
