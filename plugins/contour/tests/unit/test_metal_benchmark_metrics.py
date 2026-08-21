from __future__ import annotations

import numpy as np

from contour.domain import PolygonData
from scripts.benchmark_metal_segmentation import _rasterize_polygon_labels, measure_segmentation


def test_component_metrics_penalize_detached_background_response() -> None:
    expected = np.zeros((40, 60), dtype=np.int32)
    expected[10:30, 8:22] = 1
    predicted = np.zeros(expected.shape, dtype=np.uint8)
    predicted[10:30, 8:22] = 255
    predicted[4:10, 45:52] = 255

    metrics = measure_segmentation(predicted, expected, elapsed_ms=0.0)

    assert metrics.expected_components == 1
    assert metrics.predicted_components == 2
    assert metrics.false_positive_components == 1
    assert metrics.missed_expected_components == 0
    assert metrics.component_count_absolute_error == 1
    assert metrics.component_precision == 0.5
    assert metrics.component_recall == 1.0
    assert metrics.component_f1 == 2.0 / 3.0


def test_component_metrics_penalize_a_missed_conductor() -> None:
    expected = np.zeros((40, 60), dtype=np.int32)
    expected[8:18, 5:25] = 1
    expected[24:34, 35:55] = 2
    predicted = np.zeros(expected.shape, dtype=np.uint8)
    predicted[8:18, 5:25] = 255

    metrics = measure_segmentation(predicted, expected, elapsed_ms=0.0)

    assert metrics.matched_expected_components == 1
    assert metrics.missed_expected_components == 1
    assert metrics.component_recall == 0.5


def test_component_metrics_penalize_a_false_merge() -> None:
    expected = np.zeros((40, 60), dtype=np.int32)
    expected[10:30, 5:25] = 1
    expected[10:30, 35:55] = 2
    predicted = np.zeros(expected.shape, dtype=np.uint8)
    predicted[10:30, 5:55] = 255

    metrics = measure_segmentation(predicted, expected, elapsed_ms=0.0)

    assert metrics.predicted_components == 1
    assert metrics.false_merges == 1
    assert metrics.false_positive_components == 0


def test_component_metrics_count_small_objects_inside_one_large_merge() -> None:
    expected = np.zeros((40, 100), dtype=np.int32)
    for object_id, x_start in enumerate(range(5, 95, 10), start=1):
        expected[15:25, x_start : x_start + 5] = object_id
    predicted = np.zeros(expected.shape, dtype=np.uint8)
    predicted[10:30, 2:98] = 255

    metrics = measure_segmentation(predicted, expected, elapsed_ms=0.0)

    assert metrics.false_merges == 8
    assert metrics.false_positive_components == 0


def test_component_metrics_preserve_polygon_identity_across_a_hole_cut() -> None:
    expected = np.zeros((40, 60), dtype=np.int32)
    expected[5:35, 10:50] = 1
    expected[18:22, 10:50] = 0
    predicted_labels = expected.copy()
    predicted = np.where(predicted_labels > 0, 255, 0).astype(np.uint8)

    metrics = measure_segmentation(
        predicted,
        expected,
        elapsed_ms=0.0,
        predicted_labels=predicted_labels,
    )

    assert metrics.expected_components == 1
    assert metrics.predicted_components == 1
    assert metrics.false_splits == 0
    assert metrics.component_f1 == 1.0


def test_rasterization_preserves_independent_conductor_inside_parent_hole() -> None:
    polygons = [
        PolygonData(
            id=1,
            points=[(2, 2), (37, 2), (37, 37), (2, 37)],
        ),
        PolygonData(
            id=2,
            points=[(8, 8), (31, 8), (31, 31), (8, 31)],
            is_hole=True,
            parent_id=1,
        ),
        PolygonData(
            id=3,
            points=[(14, 14), (25, 14), (25, 25), (14, 25)],
        ),
    ]

    labels = _rasterize_polygon_labels(polygons, (40, 40))

    assert labels[4, 4] > 0
    assert labels[10, 10] == 0
    assert labels[20, 20] > 0
    assert labels[20, 20] != labels[4, 4]
