"""One scale for every score of every analysis mode: which way is better, how good a value is, its
word, its colour, how it is written and where a frame stands among the others.

Before, the direction of a metric was told in two places that disagreed (the colour said a frame was
good while the word said it was poor), every window kept its own colours, a run of equal values got
percentiles from 0 to 100, and a missing value counted as zero. Everything that shows a score asks
here instead:

- ``metric_higher_is_better`` — the one answer to "is more better?";
- ``metric_visual_ratio`` — where the value lies on the metric's own range (0..1);
- ``metric_goodness`` — the same turned so that 1 is always good and 0 always bad;
- ``goodness_level_key`` / ``metric_level_key`` — good, fair or poor, from the goodness;
- ``goodness_colors`` / ``score_badge_style`` — the colour of that level, the same in every window;
- ``format_metric_value`` — how the value is written (a 0–100 score, a share in %, a raw number);
- ``goodness_percentiles`` — the share of frames that are worse, equal values sharing one percentile.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

# --- Direction -----------------------------------------------------------------------------------

# Plain metrics where more is better; every other plain metric is a risk, an error or a distance.
HIGHER_IS_BETTER_KEYS = frozenset(
    {
        "model_model_score",
        "iou",
        "dice",
        "iou_score",
        "dice_score",
        "polygon_bce_score",
        "overall_polygon_score",
        "precision",
        "recall",
        "f1",
        "precision_score",
        "recall_score",
        "f1_score",
        "localization_score",
        "overall_point_score",
        "confidence_model_score",
        "confidence_difference_score",
        "confidence_bce_score",
        "confidence_threshold_crossing_score",
        "grid_damage_quality",
    }
)
# Per-model metric families (family::model) where more is better.
HIGHER_IS_BETTER_FAMILIES = frozenset({"model_point_contrast"})
# Operations of model pairs where more is better.
HIGHER_IS_BETTER_PAIR_OPERATIONS = frozenset({"iou", "dice"})
HIGHER_IS_BETTER_CONFIDENCE_PAIR_OPERATIONS = frozenset({"correlation", "low_iou"})

# --- Ranges --------------------------------------------------------------------------------------

# Scores and risks on 0..100.
SCORE_100_KEYS = frozenset(
    {
        "topology_risk",
        "photo_mismatch_risk",
        "photo_quality_risk",
        "overall_polygon_score",
        "iou_score",
        "dice_score",
        "polygon_bce_score",
        "overall_point_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "localization_score",
        "confidence_model_score",
        "confidence_difference_score",
        "confidence_bce_score",
        "confidence_threshold_crossing_score",
        "single_result_risk_score",
        "mask_structure_risk",
        "batch_outlier_risk",
        "source_alignment_risk",
        "source_mask_agreement_risk",
        "confidence_risk",
        "heuristic_check_score",
        "heuristic_gross_risk",
        "heuristic_topology_risk",
        "heuristic_contamination_risk",
        "heuristic_width_risk",
        "heuristic_geometry_risk",
        "heuristic_batch_risk",
    }
)
SCORE_100_FAMILIES = frozenset({"photo_mismatch_risk", "confidence_ece", "confident_error_share"})
# Counts: this many is as bad as it gets on the colour scale.
COUNT_KEYS: dict[str, float] = {"topology_events": 10.0}
DISTANCE_KEYS = frozenset({"mean_localization_distance"})
BCE_KEYS = frozenset({"bce"})
# Uncertainty of one model's confidence: a little is normal, so the bands sit higher.
CONFIDENCE_FAMILIES = frozenset({"model_confidence", "model_output_confidence"})

# Goodness at and above which a value is fair, and good (below the first it is poor).
DEFAULT_LEVEL_BREAKS = (0.33, 0.66)
CONFIDENCE_LEVEL_BREAKS = (0.40, 0.65)
# The overall frame risk and the model agreement blend many similarities of two masks, so their range is
# narrow: 1 - agreement of 0.25 is already two models with IoU about 55%. Risk up to 7% is good, up to
# 20% fair, above it poor (checked on the snapshot: IoU 99.9% — 0.6%, IoU 83% — 9%, IoU 55% — 25%).
AGREEMENT_KEYS = frozenset({"overall_frame_score", "export_priority_score", "disagreement_score", "model_model_score"})
AGREEMENT_LEVEL_BREAKS = (0.80, 0.93)

LEVEL_GOOD = "score.level.good"
LEVEL_FAIR = "score.level.fair"
LEVEL_POOR = "score.level.poor"

# The colours of the levels, one set for the whole program: (background, text).
LEVEL_COLORS: dict[str | None, tuple[str, str]] = {
    LEVEL_GOOD: ("#1f5f3b", "#e9fff1"),
    LEVEL_FAIR: ("#8a6a12", "#fff7da"),
    LEVEL_POOR: ("#8c2f39", "#ffe9ec"),
    None: ("#2f3844", "#edf3fb"),
}
LEVEL_RGB: dict[str | None, tuple[int, int, int]] = {
    LEVEL_GOOD: (31, 95, 59),
    LEVEL_FAIR: (138, 106, 18),
    LEVEL_POOR: (140, 47, 57),
    None: (47, 56, 68),
}


def _split(metric_key: str | None) -> tuple[str, list[str]]:
    parts = str(metric_key or "").split("::")
    return parts[0], parts[1:]


def metric_family(metric_key: str | None) -> str:
    return _split(metric_key)[0]


def metric_higher_is_better(metric_key: str | None) -> bool:
    """Whether a bigger value of the metric is a better frame. The one place that says it."""

    head, rest = _split(metric_key)
    if head == "pair" and len(rest) == 3:
        return rest[2].lower() in HIGHER_IS_BETTER_PAIR_OPERATIONS
    if head == "confidence_pair" and len(rest) == 3:
        return rest[2].lower() in HIGHER_IS_BETTER_CONFIDENCE_PAIR_OPERATIONS
    if head == "combined_pair":
        return False
    if rest:
        return head in HIGHER_IS_BETTER_FAMILIES
    return head in HIGHER_IS_BETTER_KEYS


def metric_is_lower_better(metric_key: str | None) -> bool:
    return not metric_higher_is_better(metric_key)


def metric_uses_100_scale(metric_key: str | None) -> bool:
    key = str(metric_key or "")
    return key in SCORE_100_KEYS or ("::" in key and metric_family(key) in SCORE_100_FAMILIES)


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def metric_visual_ratio(
    metric_key: str | None,
    value: float | None,
    *,
    point_match_radius: float = 3.0,
    bce_score_cap: float = 1.0,
) -> float | None:
    """Where the value lies on the metric's own range, 0..1 (not yet turned to "good")."""

    numeric = _finite(value)
    if numeric is None:
        return None
    key = str(metric_key or "")
    if metric_uses_100_scale(key):
        ratio = numeric / 100.0
    elif key in BCE_KEYS:
        ratio = numeric / max(1e-9, float(bce_score_cap))
    elif key in DISTANCE_KEYS:
        ratio = numeric / max(1e-9, float(point_match_radius))
    elif key in COUNT_KEYS:
        ratio = numeric / COUNT_KEYS[key]
    else:
        ratio = numeric
    return max(0.0, min(float(ratio), 1.0))


def metric_goodness(
    metric_key: str | None,
    value: float | None,
    *,
    point_match_radius: float = 3.0,
    bce_score_cap: float = 1.0,
) -> float | None:
    """How good the value is, 0 (bad) .. 1 (good), whichever way the metric runs.

    It is calibrated so that the same goodness means the same thing for every metric: below 0.33 poor,
    from 0.66 good (a metric with its own breaks is stretched onto these), so the colour of a matrix
    tile and the word of a badge always agree.
    """

    ratio = metric_visual_ratio(metric_key, value, point_match_radius=point_match_radius, bce_score_cap=bce_score_cap)
    if ratio is None:
        return None
    raw = ratio if metric_higher_is_better(metric_key) else 1.0 - ratio
    return _calibrated(raw, level_breaks(metric_key))


def _calibrated(raw: float, breaks: tuple[float, float]) -> float:
    fair_from, good_from = (float(value) for value in breaks)
    target_fair, target_good = DEFAULT_LEVEL_BREAKS
    if (fair_from, good_from) == (target_fair, target_good):
        return raw
    if raw < fair_from:
        return target_fair * raw / max(1e-9, fair_from)
    if raw < good_from:
        return target_fair + (target_good - target_fair) * (raw - fair_from) / max(1e-9, good_from - fair_from)
    return target_good + (1.0 - target_good) * (raw - good_from) / max(1e-9, 1.0 - good_from)


def level_breaks(metric_key: str | None) -> tuple[float, float]:
    if str(metric_key or "") in AGREEMENT_KEYS:
        return AGREEMENT_LEVEL_BREAKS
    return CONFIDENCE_LEVEL_BREAKS if metric_family(metric_key) in CONFIDENCE_FAMILIES else DEFAULT_LEVEL_BREAKS


def goodness_level_key(goodness: float | None, breaks: tuple[float, float] = DEFAULT_LEVEL_BREAKS) -> str | None:
    numeric = _finite(goodness)
    if numeric is None:
        return None
    fair_from, good_from = breaks
    if numeric >= good_from:
        return LEVEL_GOOD
    if numeric >= fair_from:
        return LEVEL_FAIR
    return LEVEL_POOR


def metric_level_key(
    metric_key: str | None,
    value: float | None,
    *,
    point_match_radius: float = 3.0,
    bce_score_cap: float = 1.0,
) -> str | None:
    """Good, fair or poor — the word always agrees with the colour (both come from the goodness)."""

    goodness = metric_goodness(metric_key, value, point_match_radius=point_match_radius, bce_score_cap=bce_score_cap)
    return goodness_level_key(goodness)


def goodness_colors(level_key: str | None) -> tuple[str, str]:
    return LEVEL_COLORS.get(level_key, LEVEL_COLORS[None])


def score_badge_style(level_key: str | None, *, padding: str = "6px 10px", radius: int = 8) -> str:
    background, foreground = goodness_colors(level_key)
    return (
        f"padding: {padding}; border-radius: {int(radius)}px; background-color: {background}; "
        f"color: {foreground}; font-weight: 700;"
    )


def metric_badge_style(
    metric_key: str | None,
    value: float | None,
    *,
    point_match_radius: float = 3.0,
    bce_score_cap: float = 1.0,
    padding: str = "6px 10px",
    radius: int = 8,
) -> str:
    level = metric_level_key(metric_key, value, point_match_radius=point_match_radius, bce_score_cap=bce_score_cap)
    return score_badge_style(level, padding=padding, radius=radius)


def percentile_level_key(percentile: float | None) -> str | None:
    numeric = _finite(percentile)
    return None if numeric is None else goodness_level_key(max(0.0, min(numeric, 100.0)) / 100.0)


def metric_value_kind(metric_key: str | None) -> str:
    """How a value is written: ``score`` (0..100), ``share`` (0..1 shown in %), ``bce``, ``distance``, ``count``."""

    key = str(metric_key or "")
    if metric_uses_100_scale(key):
        return "score"
    if key in BCE_KEYS:
        return "bce"
    if key in DISTANCE_KEYS:
        return "distance"
    if key in COUNT_KEYS:
        return "count"
    return "share"


def format_metric_value(metric_key: str | None, value: float | None) -> str:
    numeric = _finite(value)
    if numeric is None:
        return "-"
    kind = metric_value_kind(metric_key)
    if kind == "score":
        return f"{numeric:.1f}"
    if kind == "share":
        return f"{numeric * 100.0:.1f}%"
    if kind == "distance":
        return f"{numeric:.2f} px"
    if kind == "count":
        return f"{numeric:.0f}"
    return f"{numeric:.4f}"


# --- Where a frame stands among the others -------------------------------------------------------


def goodness_percentiles(values: Mapping[str, float], metric_key: str | None) -> dict[str, float]:
    """Goodness percentile of every frame: 100 — better than all the others, 0 — worse than all.

    Frames with equal values share one percentile (the middle of the places they hold), so a run of
    equal values does not spread from red to green; one frame, or all frames equal, stand at 50.
    """

    items = [(str(key), float(value)) for key, value in values.items() if _finite(value) is not None]
    if not items:
        return {}
    higher = metric_higher_is_better(metric_key)
    # Worst first.
    items.sort(key=lambda item: item[1], reverse=not higher)
    count = len(items)
    if count == 1:
        return {items[0][0]: 50.0}
    result: dict[str, float] = {}
    start = 0
    while start < count:
        end = start
        base = items[start][1]
        while end + 1 < count and math.isclose(items[end + 1][1], base, rel_tol=1e-9, abs_tol=1e-12):
            end += 1
        # Places start..end are tied: they share the middle place.
        middle = (start + end) / 2.0
        for index in range(start, end + 1):
            result[items[index][0]] = 100.0 * middle / (count - 1)
        start = end + 1
    return result


def relative_goodness(values: Mapping[str, float], metric_key: str | None, **scale) -> dict[str, float]:
    """Goodness of every frame stretched between the worst and the best of the run (0 worst, 1 best).

    When all values are equal there is nothing to stretch: the absolute goodness is used instead.
    """

    finite = {str(key): float(value) for key, value in values.items() if _finite(value) is not None}
    if not finite:
        return {}
    low, high = min(finite.values()), max(finite.values())
    if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-12):
        return {key: float(metric_goodness(metric_key, value, **scale) or 0.0) for key, value in finite.items()}
    higher = metric_higher_is_better(metric_key)
    span = high - low
    return {key: ((value - low) / span if higher else (high - value) / span) for key, value in finite.items()}


def worst_first(values: Iterable[tuple[str, float]], metric_key: str | None) -> list[str]:
    higher = metric_higher_is_better(metric_key)
    return [key for key, _value in sorted(values, key=lambda item: item[1], reverse=not higher)]


__all__ = [
    "LEVEL_COLORS",
    "LEVEL_FAIR",
    "LEVEL_GOOD",
    "LEVEL_POOR",
    "LEVEL_RGB",
    "format_metric_value",
    "goodness_colors",
    "goodness_level_key",
    "goodness_percentiles",
    "level_breaks",
    "metric_badge_style",
    "metric_family",
    "metric_goodness",
    "metric_higher_is_better",
    "metric_is_lower_better",
    "metric_level_key",
    "metric_uses_100_scale",
    "metric_value_kind",
    "metric_visual_ratio",
    "percentile_level_key",
    "relative_goodness",
    "score_badge_style",
    "worst_first",
]
