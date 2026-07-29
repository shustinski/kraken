"""Pure threshold fitting for heuristic contact-recognition feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from ...vision.via_detection import ViaDetection
from ..processing import ContourExtractionSettings


@dataclass(frozen=True, slots=True)
class ContactFeedbackChange:
    field: str
    feature: str
    old_value: float
    new_value: float


@dataclass(frozen=True, slots=True)
class ContactFeedbackAdjustment:
    changes: tuple[ContactFeedbackChange, ...] = ()
    selected_feature: str | None = None
    reason: str = ""
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ThresholdSpec:
    field: str
    feature: str
    direction: Literal["min", "max"]
    domain_min: float
    domain_max: float
    step: float


# The order is also the deterministic tie-break priority for negative feedback.
_BASE_SPECS = (
    _ThresholdSpec("heuristic_min_circularity", "circularity", "min", 0.0, 1.0, 0.001),
    _ThresholdSpec("heuristic_min_compactness", "compactness", "min", 0.0, 1.0, 0.001),
    _ThresholdSpec("heuristic_min_center_contrast", "contrast", "min", 0.0, 255.0, 0.1),
    _ThresholdSpec("heuristic_min_peak_prominence", "prominence", "min", 0.0, 255.0, 0.1),
    _ThresholdSpec("heuristic_min_edge_sharpness", "edge_sharpness", "min", 0.0, 10.0, 0.01),
    _ThresholdSpec("heuristic_max_elongation", "aspect", "max", 1.0, 20.0, 0.01),
    _ThresholdSpec("heuristic_max_line_coherence", "line_coherence", "max", 0.0, 1.0, 0.01),
    _ThresholdSpec("heuristic_min_center_brightness", "center_brightness", "min", 0.0, 255.0, 0.1),
    _ThresholdSpec("heuristic_max_center_drift_ratio", "center_drift_ratio", "max", 0.1, 1.5, 0.01),
)


def _feature_value(detection: ViaDetection, feature: str) -> float:
    if feature == "score":
        return float(detection.score)
    if feature == "contrast":
        return float(detection.contrast)
    if feature == "prominence":
        return float(detection.prominence)
    if feature == "compactness":
        return float(detection.compactness)
    if feature == "aspect":
        return float(detection.aspect)
    if feature == "center_drift_ratio":
        diameter = max(1e-9, float(detection.features.get("diameter", detection.diameter_estimate)))
        return float(detection.features.get("center_drift", 0.0)) / diameter
    return float(detection.features.get(feature, 0.0))


def _quantize_outward(value: float, step: float, direction: Literal["min", "max"]) -> float:
    scaled = float(value) / float(step)
    units = math.floor(scaled + 1e-9) if direction == "min" else math.ceil(scaled - 1e-9)
    return float(units * step)


def fit_positive_contact(
    settings: ContourExtractionSettings,
    detection: ViaDetection | None,
) -> ContactFeedbackAdjustment:
    """Expand only threshold boundaries violated by a manually added contact."""

    if detection is None:
        return ContactFeedbackAdjustment(reason="measurement_failed")
    changes: list[ContactFeedbackChange] = []
    specs = _BASE_SPECS
    seen_fields: set[str] = set()
    for spec in specs:
        if spec.field in seen_fields:
            continue
        seen_fields.add(spec.field)
        old_value = float(getattr(settings, spec.field))
        measured = _feature_value(detection, spec.feature)
        violated = measured < old_value if spec.direction == "min" else measured > old_value
        if not violated:
            continue
        new_value = _quantize_outward(measured, spec.step, spec.direction)
        new_value = min(spec.domain_max, max(spec.domain_min, new_value))
        if new_value != old_value:
            changes.append(ContactFeedbackChange(spec.field, spec.feature, old_value, new_value))
    return ContactFeedbackAdjustment(
        changes=tuple(changes),
        reason="" if changes else "already_within_thresholds",
        diagnostics={"measured_score": float(detection.score)},
    )


def fit_negative_contact(
    settings: ContourExtractionSettings,
    removed: ViaDetection | None,
    references: list[ViaDetection],
) -> ContactFeedbackAdjustment:
    """Tighten one most discriminating threshold while retaining all references."""

    if removed is None:
        return ContactFeedbackAdjustment(reason="measurement_failed")
    if len(references) < 2:
        return ContactFeedbackAdjustment(reason="insufficient_references")

    candidates: list[tuple[float, int, _ThresholdSpec, float, float, float]] = []
    specs = _BASE_SPECS
    for priority, spec in enumerate(specs):
        removed_value = _feature_value(removed, spec.feature)
        values = [_feature_value(item, spec.feature) for item in references]
        reference_median = float(median(values))
        if spec.direction == "min":
            if not all(value > removed_value for value in values):
                continue
            nearest = min(values)
            new_value = _quantize_outward(0.5 * (removed_value + nearest), spec.step, "max")
            if new_value <= float(getattr(settings, spec.field)):
                continue
        else:
            if not all(value < removed_value for value in values):
                continue
            nearest = max(values)
            new_value = _quantize_outward(0.5 * (removed_value + nearest), spec.step, "min")
            if new_value >= float(getattr(settings, spec.field)):
                continue
        new_value = min(spec.domain_max, max(spec.domain_min, new_value))
        relative_gap = abs(removed_value - reference_median) / max(
            1e-9, spec.domain_max - spec.domain_min
        )
        candidates.append((relative_gap, priority, spec, new_value, removed_value, reference_median))

    if not candidates:
        return ContactFeedbackAdjustment(reason="no_separating_threshold")
    relative_gap, _priority, spec, new_value, removed_value, reference_median = min(
        candidates, key=lambda item: (-item[0], item[1])
    )
    old_value = float(getattr(settings, spec.field))
    return ContactFeedbackAdjustment(
        changes=(ContactFeedbackChange(spec.field, spec.feature, old_value, new_value),),
        selected_feature=spec.feature,
        diagnostics={
            "removed_value": removed_value,
            "reference_median": reference_median,
            "relative_gap": relative_gap,
            "reference_count": float(len(references)),
        },
    )
