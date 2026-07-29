from __future__ import annotations

import random

import numpy as np

from contour.vision.via_detection.config import TemplateViaDetectorConfig
from contour.vision.via_detection.result import ViaDetection
from contour.vision.via_detection.template_detector import (
    TemplateRawMatch,
    _collect_peaks,
    _suppress_duplicate_matches,
    score_vias_template_raw,
)


def _match(x: float, y: float, size: int, score: float) -> tuple[TemplateRawMatch, ViaDetection]:
    raw = TemplateRawMatch(
        x=x,
        y=y,
        bbox=(round(x), round(y), size, size),
        score=score,
        diameter_estimate=float(size),
    )
    detection = ViaDetection(
        x=x,
        y=y,
        bbox=raw.bbox,
        score=score,
        diameter_estimate=float(size),
        contrast=0.0,
        prominence=0.0,
        compactness=0.0,
        aspect=1.0,
    )
    return raw, detection


def _quadratic_suppression(
    eligible: list[tuple[TemplateRawMatch, ViaDetection]],
) -> list[tuple[TemplateRawMatch, ViaDetection]]:
    kept: list[tuple[TemplateRawMatch, ViaDetection]] = []
    for raw, detection in eligible:
        distance = max(raw.bbox[2], raw.bbox[3]) + 2
        if any(
            (detection.x - other.x) ** 2 + (detection.y - other.y) ** 2
            <= float(max(distance, max(other_raw.bbox[2], other_raw.bbox[3]) + 2) ** 2)
            for other_raw, other in kept
        ):
            continue
        kept.append((raw, detection))
    return kept


def test_spatial_template_suppression_matches_original_variable_radius_rule() -> None:
    randomizer = random.Random(27)
    eligible = [
        _match(
            randomizer.uniform(0.0, 800.0),
            randomizer.uniform(0.0, 600.0),
            randomizer.randint(5, 21),
            100.0 - index * 0.01,
        )
        for index in range(800)
    ]

    expected = _quadratic_suppression(eligible)
    actual = _suppress_duplicate_matches(eligible)

    assert [raw for raw, _detection in actual] == [raw for raw, _detection in expected]


def test_peak_collection_keeps_all_accepted_and_caps_debug_only_candidates() -> None:
    response = np.zeros((80, 80), dtype=np.float32)
    accepted_points = {(5, 5), (15, 15), (25, 25)}
    for y_coord, x_coord in accepted_points:
        response[y_coord, x_coord] = 0.9
    for index, (y_coord, x_coord) in enumerate(
        (divmod(value, 8) for value in range(64)),
        start=1,
    ):
        response[y_coord * 9 + 4, x_coord * 9 + 4] = 0.1 + index * 0.005
    output: list[TemplateRawMatch] = []

    _collect_peaks(
        response,
        np.ones((5, 5), dtype=np.uint8),
        0.8,
        output,
        template_index=0,
        below_threshold_limit=7,
    )

    accepted = {
        (round(match.y - 2.5), round(match.x - 2.5))
        for match in output
        if match.score >= 80.0
    }
    assert accepted_points <= accepted
    assert sum(match.score < 80.0 for match in output) == 7


def test_template_result_caps_below_threshold_debug_without_capping_acceptance() -> None:
    accepted = [_match(float(index * 20), 10.0, 5, 90.0)[0] for index in range(12)]
    below = [_match(float(index), 30.0, 5, 20.0)[0] for index in range(12_500)]

    result = score_vias_template_raw(
        [*accepted, *below],
        (100, 20_000),
        TemplateViaDetectorConfig(
            templates=[np.ones((5, 5), dtype=np.uint8)],
            min_correlation=0.8,
        ),
    )

    assert len(result.accepted) == len(accepted)
    assert len(result.below_threshold) == 10_000
    assert result.parameters_snapshot["below_threshold_count"] == len(below)
