from __future__ import annotations

from polygon_widget.application.services.quality_gates import (
    SemQualityGateThresholds,
    SemQualityMetrics,
    evaluate_sem_quality_gates,
)


def test_quality_gates_pass_for_good_metrics() -> None:
    metrics = SemQualityMetrics(
        via_precision=0.92,
        via_recall=0.86,
        conductor_recall=0.94,
        via_false_positives_per_frame=0.6,
        via_center_shift_px=1.3,
    )
    passed, failures = evaluate_sem_quality_gates(metrics)
    assert passed is True
    assert failures == []


def test_quality_gates_return_failed_metrics() -> None:
    metrics = SemQualityMetrics(
        via_precision=0.70,
        via_recall=0.79,
        conductor_recall=0.82,
        via_false_positives_per_frame=2.2,
        via_center_shift_px=2.8,
    )
    passed, failures = evaluate_sem_quality_gates(
        metrics, thresholds=SemQualityGateThresholds(via_recall_min=0.8)
    )
    assert passed is False
    assert failures == [
        "via_precision",
        "via_recall",
        "conductor_recall",
        "via_false_positives_per_frame",
        "via_center_shift_px",
    ]
