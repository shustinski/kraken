"""benchmark_metal_segmentation: topology errors."""
from __future__ import annotations
import cv2
import numpy as np

def _topology_errors(
    predicted: np.ndarray,
    expected_labels: np.ndarray,
    predicted_labels: np.ndarray | None = None,
) -> tuple[int, int, int, int, int, int]:
    if predicted_labels is None:
        _count, predicted_labels = cv2.connectedComponents(
            (predicted > 0).astype(np.uint8),
            connectivity=8,
        )
    predicted_ids = np.unique(predicted_labels[predicted_labels > 0])
    expected_ids = np.unique(expected_labels[expected_labels > 0])
    expected_areas = np.bincount(expected_labels.ravel())
    predicted_areas = np.bincount(predicted_labels.ravel())
    supported_predictions = 0
    false_merges = 0
    for predicted_id in predicted_ids:
        overlaps = expected_labels[predicted_labels == predicted_id]
        object_ids = np.unique(overlaps[overlaps > 0])
        prediction_area = max(1, int(overlaps.size))
        supported_predictions += int(np.count_nonzero(overlaps > 0) / prediction_area >= 0.10)
        materially_covered_objects = 0
        for object_id in object_ids:
            intersection = int(np.count_nonzero(overlaps == object_id))
            expected_area = max(1, int(expected_areas[int(object_id)]))
            if intersection / expected_area >= 0.10:
                materially_covered_objects += 1
        false_merges += max(0, materially_covered_objects - 1)

    false_splits = 0
    matched_expected = 0
    for object_id in expected_ids:
        overlaps = predicted_labels[expected_labels == object_id]
        expected_area = max(1, int(overlaps.size))
        matched_expected += int(np.count_nonzero(overlaps > 0) / expected_area >= 0.50)
        component_ids = np.unique(overlaps[overlaps > 0])
        material_overlaps = 0
        for component_id in component_ids:
            intersection = int(np.count_nonzero(overlaps == component_id))
            prediction_area = max(1, int(predicted_areas[int(component_id)]))
            if intersection / expected_area >= 0.01 and intersection / prediction_area >= 0.10:
                material_overlaps += 1
        false_splits += max(0, material_overlaps - 1)
    predicted_components = int(predicted_ids.size)
    false_positive_components = predicted_components - supported_predictions
    missed_expected_components = int(expected_ids.size) - matched_expected
    return (
        predicted_components,
        false_merges,
        false_splits,
        false_positive_components,
        missed_expected_components,
        matched_expected,
    )
