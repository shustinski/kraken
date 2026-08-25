from __future__ import annotations

import inspect

import numpy as np
import pytest

from contour.vision.metal_recovery import MetalRecoveryConfig
from scripts.benchmark_metal_segmentation import (
    BenchmarkCase,
    crop_evaluation_region,
    measure_segmentation,
    prepare_evaluation_masks,
    run_benchmark,
)

CROP_PX = 50
SHAPE = (200, 200)


def _blank_labels() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.int32)


def _evaluate(
    predicted: np.ndarray,
    expected: np.ndarray,
    crop_px: int,
    predicted_labels: np.ndarray | None = None,
):
    predicted_eval, expected_eval, predicted_labels_eval = prepare_evaluation_masks(
        predicted,
        expected,
        crop_px=crop_px,
        predicted_labels=predicted_labels,
    )
    return measure_segmentation(
        predicted_eval,
        expected_eval,
        elapsed_ms=0.0,
        predicted_labels=predicted_labels_eval,
    )


def test_object_fully_inside_discarded_border_does_not_affect_metrics() -> None:
    expected = _blank_labels()
    expected[80:120, 80:120] = 1
    expected[8:22, 8:22] = 2
    predicted = np.where(expected > 0, 255, 0).astype(np.uint8)

    center_only_expected = _blank_labels()
    center_only_expected[80:120, 80:120] = 1
    center_only_predicted = np.where(center_only_expected > 0, 255, 0).astype(np.uint8)

    cropped = _evaluate(predicted, expected, CROP_PX)
    center_only = _evaluate(center_only_predicted, center_only_expected, CROP_PX)

    assert cropped.iou == pytest.approx(center_only.iou)
    assert cropped.precision == pytest.approx(center_only.precision)
    assert cropped.recall == pytest.approx(center_only.recall)
    assert cropped.boundary_f1 == pytest.approx(center_only.boundary_f1)
    assert cropped.expected_components == 1
    assert cropped.predicted_components == 1
    assert cropped.false_positive_components == 0
    assert cropped.missed_expected_components == 0
    assert cropped.false_merges == 0
    assert cropped.false_splits == 0
    assert cropped.topology_exact_match is True


def test_object_crossing_crop_boundary_remains_a_normal_component() -> None:
    expected = _blank_labels()
    expected[0:80, 90:110] = 1
    predicted = np.where(expected > 0, 255, 0).astype(np.uint8)

    metrics = _evaluate(predicted, expected, CROP_PX)

    assert metrics.expected_components == 1
    assert metrics.predicted_components == 1
    assert metrics.false_positive_components == 0
    assert metrics.missed_expected_components == 0
    assert metrics.topology_exact_match is True
    predicted_eval = crop_evaluation_region(predicted, CROP_PX)
    assert predicted_eval[0, 40] > 0
    assert int(np.count_nonzero(predicted_eval)) == 30 * 20


def test_matching_object_crossing_crop_boundary_is_topology_match() -> None:
    expected = _blank_labels()
    expected[0:90, 70:130] = 1
    predicted = np.where(expected > 0, 255, 0).astype(np.uint8)

    metrics = _evaluate(predicted, expected, CROP_PX)

    assert metrics.iou == pytest.approx(1.0)
    assert metrics.topology_exact_match is True


def test_extra_prediction_only_in_discarded_border_is_not_a_false_component() -> None:
    expected = _blank_labels()
    expected[80:120, 80:120] = 1
    predicted = np.where(expected > 0, 255, 0).astype(np.uint8)
    predicted[6:18, 160:190] = 255

    full_frame = measure_segmentation(predicted, expected, elapsed_ms=0.0)
    cropped = _evaluate(predicted, expected, CROP_PX)

    assert full_frame.false_positive_components == 1
    assert full_frame.topology_exact_match is False
    assert cropped.false_positive_components == 0
    assert cropped.missed_expected_components == 0
    assert cropped.false_merges == 0
    assert cropped.false_splits == 0
    assert cropped.predicted_components == cropped.expected_components == 1
    assert cropped.topology_exact_match is True
    assert cropped.iou == pytest.approx(1.0)


def test_extra_prediction_inside_evaluation_roi_is_an_ordinary_error() -> None:
    expected = _blank_labels()
    expected[80:120, 80:120] = 1
    predicted = np.where(expected > 0, 255, 0).astype(np.uint8)
    predicted[120:150, 80:100] = 255

    metrics = _evaluate(predicted, expected, CROP_PX)

    assert metrics.iou < 1.0
    assert metrics.precision < 1.0
    assert metrics.false_positive_area_fraction > 0.0
    assert metrics.topology_exact_match is True
    assert metrics.predicted_components == 1
    assert metrics.expected_components == 1


def test_extra_disconnected_prediction_inside_evaluation_roi_is_a_false_component() -> None:
    expected = _blank_labels()
    expected[80:120, 80:120] = 1
    predicted = np.where(expected > 0, 255, 0).astype(np.uint8)
    predicted[130:150, 80:100] = 255

    metrics = _evaluate(predicted, expected, CROP_PX)

    assert metrics.false_positive_components == 1
    assert metrics.predicted_components == 2
    assert metrics.expected_components == 1
    assert metrics.iou < 1.0
    assert metrics.topology_exact_match is False


def test_crop_zero_matches_full_frame_methodology() -> None:
    expected = np.zeros((40, 60), dtype=np.int32)
    expected[10:30, 8:22] = 1
    predicted = np.zeros(expected.shape, dtype=np.uint8)
    predicted[10:30, 8:22] = 255
    predicted[4:10, 45:52] = 255

    baseline = measure_segmentation(predicted, expected, elapsed_ms=0.0)
    cropped = _evaluate(predicted, expected, 0)

    assert cropped == baseline
    predicted_eval, expected_eval, predicted_labels_eval = prepare_evaluation_masks(
        predicted,
        expected,
        crop_px=0,
    )
    assert predicted_labels_eval is None
    assert np.array_equal(predicted_eval, predicted)
    assert np.array_equal(expected_eval, expected)


def test_crop_zero_preserves_polygon_ids_across_a_hole_cut() -> None:
    expected = np.zeros((40, 60), dtype=np.int32)
    expected[5:35, 10:50] = 1
    expected[18:22, 10:50] = 0
    predicted_labels = expected.copy()
    predicted = np.where(predicted_labels > 0, 255, 0).astype(np.uint8)

    baseline = measure_segmentation(
        predicted,
        expected,
        elapsed_ms=0.0,
        predicted_labels=predicted_labels,
    )
    cropped = _evaluate(predicted, expected, 0, predicted_labels=predicted_labels)

    assert cropped == baseline
    assert cropped.expected_components == 1
    assert cropped.predicted_components == 1
    assert cropped.topology_exact_match is True


def test_undersized_frame_fails_instead_of_shrinking_crop() -> None:
    tiny = np.zeros((80, 80), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small for evaluation_border_crop_px=50"):
        crop_evaluation_region(tiny, CROP_PX, frame_id="tiny")

    labels = np.zeros((80, 80), dtype=np.int32)
    case = BenchmarkCase("tiny", "scale", tiny, labels)
    with pytest.raises(ValueError, match="Frame tiny: image size 80x80 is too small"):
        run_benchmark(
            ["legacy_otsu"],
            cases=[case],
            evaluation_stage="segmentation",
            evaluation_border_crop_px=CROP_PX,
        )


def test_evaluation_crop_stays_out_of_recognition_config() -> None:
    assert "evaluation_border_crop_px" not in inspect.signature(measure_segmentation).parameters
    assert "evaluation_border_crop_px" not in MetalRecoveryConfig.__dataclass_fields__
