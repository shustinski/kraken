import numpy as np

from neuralimage.metrics.segmentation import compute_segmentation_metrics


def test_segmentation_metrics_perfect_prediction():
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0
    metrics = compute_segmentation_metrics(mask, mask, threshold=0.5)
    assert metrics.dice > 0.99
    assert metrics.iou > 0.99
    assert metrics.boundary_iou > 0.9


def test_segmentation_metrics_reports_topology_counts():
    target = np.zeros((32, 32), dtype=np.float32)
    target[9:13, 4:28] = 1.0
    prediction = target.copy()
    prediction[:, 15:17] = 0.0
    metrics = compute_segmentation_metrics(prediction, target, threshold=0.5)
    assert metrics.wire_break_count == 1
    assert metrics.false_bridge_count == 0


def test_segmentation_metrics_counts_one_false_bridge():
    target = np.zeros((32, 32), dtype=np.float32)
    target[6:10, 4:28] = 1.0
    target[20:24, 4:28] = 1.0
    prediction = target.copy()
    prediction[9:21, 14:18] = 1.0
    metrics = compute_segmentation_metrics(prediction, target)
    assert metrics.false_bridge_count == 1
    assert metrics.wire_break_count == 0


def test_boundary_metrics_are_tolerance_aware_and_confidence_is_calibrated():
    target = np.zeros((32, 32), dtype=np.float32)
    target[8:24, 8:24] = 1.0
    prediction = np.roll(target, 1, axis=1)
    confidence = np.where(prediction == target, 0.95, 0.05).astype(np.float32)
    metrics = compute_segmentation_metrics(
        prediction,
        target,
        confidence=confidence,
        boundary_tolerance=2,
    )
    assert metrics.boundary_f1 > 0.99
    assert metrics.hausdorff_distance == 1.0
    assert metrics.brier_score is not None and metrics.brier_score < 0.01
    assert sum(metrics.confidence_histogram.values()) == target.size


def test_empty_masks_have_perfect_metrics_and_zero_distance():
    empty = np.zeros((12, 17), dtype=np.float32)
    metrics = compute_segmentation_metrics(empty, empty)
    assert metrics.dice == 1.0
    assert metrics.iou == 1.0
    assert metrics.boundary_iou == 1.0
    assert metrics.hausdorff_distance == 0.0
