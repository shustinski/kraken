from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SemQualityGateThresholds:
    via_precision_min: float = 0.88
    via_recall_min: float = 0.82
    conductor_recall_min: float = 0.90
    via_false_positives_per_frame_max: float = 1.2
    via_center_shift_px_max: float = 2.0


@dataclass(frozen=True, slots=True)
class SemQualityMetrics:
    via_precision: float
    via_recall: float
    conductor_recall: float
    via_false_positives_per_frame: float
    via_center_shift_px: float


def evaluate_sem_quality_gates(
    metrics: SemQualityMetrics, thresholds: SemQualityGateThresholds | None = None
) -> tuple[bool, list[str]]:
    limits = thresholds or SemQualityGateThresholds()
    failures: list[str] = []
    if metrics.via_precision < limits.via_precision_min:
        failures.append("via_precision")
    if metrics.via_recall < limits.via_recall_min:
        failures.append("via_recall")
    if metrics.conductor_recall < limits.conductor_recall_min:
        failures.append("conductor_recall")
    if metrics.via_false_positives_per_frame > limits.via_false_positives_per_frame_max:
        failures.append("via_false_positives_per_frame")
    if metrics.via_center_shift_px > limits.via_center_shift_px_max:
        failures.append("via_center_shift_px")
    return (len(failures) == 0, failures)
