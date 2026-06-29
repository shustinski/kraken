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
    target[10:12, 8:24] = 1.0
    prediction = target.copy()
    prediction[11, 16] = 0.0
    metrics = compute_segmentation_metrics(prediction, target, threshold=0.5)
    assert metrics.wire_break_count >= 0
    assert metrics.false_bridge_count >= 0
