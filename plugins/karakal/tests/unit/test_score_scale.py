"""One scale for every score: direction, level word, colour, writing and percentile agree everywhere."""

from __future__ import annotations

from pathlib import Path

import pytest

from karakal.core.analysis_modes import metric_is_lower_better as modes_lower_better
from karakal.core.analysis_modes import metric_level_key as modes_level_key
from karakal.core.domain import FrameAnalysisSummary, FrameRecord, ModelSpec
from karakal.core.metric_keys import (
    _available_metric_keys_for_models,
    combined_pair_metric_key,
    compute_metric_percentiles,
    confidence_pair_metric_key,
    metric_higher_is_better,
    pair_metric_key,
)
from karakal.core.repository_shared import CONFIDENCE_PAIR_METRIC_OPERATIONS, PAIR_METRIC_OPERATIONS
from karakal.core.score_scale import (
    LEVEL_FAIR,
    LEVEL_GOOD,
    LEVEL_POOR,
    format_metric_value,
    goodness_colors,
    goodness_percentiles,
    metric_badge_style,
    metric_goodness,
    metric_level_key,
    metric_uses_100_scale,
    metric_value_kind,
    percentile_level_key,
    relative_goodness,
)


def _all_metric_keys() -> list[str]:
    specs = (
        ModelSpec("a", "A", Path("."), prob_folder=Path(".")),
        ModelSpec("b", "B", Path("."), prob_folder=Path(".")),
    )
    records = (
        FrameRecord("k", "k", original_path="x", model_mask_paths={"a": "x", "b": "y"}, model_prob_paths={"a": "p", "b": "q"}),
    )
    keys = list(_available_metric_keys_for_models(specs, records))
    keys += [pair_metric_key("a", "b", operation) for operation in PAIR_METRIC_OPERATIONS]
    keys += [confidence_pair_metric_key("a", "b", operation) for operation in CONFIDENCE_PAIR_METRIC_OPERATIONS]
    keys += [combined_pair_metric_key("a", "b"), "single_result_risk_score", "heuristic_check_score", "grid_inspection_damage_score"]
    return sorted(set(keys))


def _extremes(key: str) -> tuple[float, float]:
    """(the best value, the worst value) of a metric on its own range."""

    kind = metric_value_kind(key)
    top = {"score": 100.0, "share": 1.0, "bce": 1.0, "distance": 3.0, "count": 10.0}[kind]
    return (top, 0.0) if metric_higher_is_better(key) else (0.0, top)


@pytest.mark.parametrize("key", _all_metric_keys())
def test_word_and_colour_follow_one_direction(key: str) -> None:
    best, worst = _extremes(key)
    assert modes_lower_better(key) is (not metric_higher_is_better(key)), "the two old names agree"
    assert metric_goodness(key, best) == pytest.approx(1.0) and metric_goodness(key, worst) == pytest.approx(0.0)
    assert metric_level_key(key, best) == LEVEL_GOOD and modes_level_key(key, best, point_match_radius=3.0, bce_score_cap=1.0) == LEVEL_GOOD
    assert metric_level_key(key, worst) == LEVEL_POOR
    assert goodness_colors(LEVEL_GOOD)[0] in metric_badge_style(key, best)
    assert goodness_colors(LEVEL_POOR)[0] in metric_badge_style(key, worst)


@pytest.mark.parametrize(
    ("key", "value", "level"),
    [
        ("overall_frame_score", 0.05, LEVEL_GOOD),  # a risk of 5%: good (it used to read "poor" in green)
        ("overall_frame_score", 0.9, LEVEL_POOR),
        ("overall_frame_score", 0.25, LEVEL_POOR),  # two models at IoU about 55%
        ("overall_frame_score", 0.09, LEVEL_FAIR),
        ("model_model_score", 0.99, LEVEL_GOOD),
        (pair_metric_key("a", "b", "xor"), 0.0, LEVEL_GOOD),
        (confidence_pair_metric_key("a", "b", "mae"), 0.0, LEVEL_GOOD),
        ("disagreement_score", 0.02, LEVEL_GOOD),
        ("polygon_bce_score", 0.0, LEVEL_POOR),
        ("bce", 0.0, LEVEL_GOOD),
        ("topology_risk", 50.0, LEVEL_FAIR),
        ("model_uncertain_fraction::a", 0.1, LEVEL_GOOD),
        ("model_confidence::a", 0.5, LEVEL_FAIR),
    ],
)
def test_levels_of_the_metrics_that_used_to_disagree(key: str, value: float, level: str) -> None:
    assert metric_level_key(key, value) == level


def test_values_are_written_one_way_per_kind() -> None:
    assert format_metric_value("iou_score", 87.345) == "87.3"
    assert format_metric_value("overall_frame_score", 0.1234) == "12.3%"
    assert format_metric_value(pair_metric_key("a", "b", "iou"), 0.5) == "50.0%"
    assert format_metric_value("bce", 0.123456) == "0.1235"
    assert format_metric_value("mean_localization_distance", 1.5) == "1.50 px"
    assert format_metric_value("topology_events", 3.0) == "3"
    assert format_metric_value("iou_score", None) == "-"
    assert metric_uses_100_scale("confidence_ece::a") and not metric_uses_100_scale("model_confidence::a")


def test_equal_values_share_one_percentile() -> None:
    values = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 50.0}
    higher = goodness_percentiles(values, "polygon_bce_score")
    assert higher["a"] == higher["b"] == higher["c"] == pytest.approx(100.0 / 3.0)
    assert higher["d"] == pytest.approx(100.0), "the only good frame is the best"
    lower = goodness_percentiles(values, "topology_risk")
    assert lower["a"] == lower["b"] == lower["c"] == pytest.approx(100.0 * 2.0 / 3.0) and lower["d"] == 0.0
    assert set(goodness_percentiles({"a": 0.0, "b": 0.0}, "bce").values()) == {50.0}
    assert goodness_percentiles({"only": 3.0}, "bce") == {"only": 50.0}
    assert percentile_level_key(goodness_percentiles(values, "polygon_bce_score")["a"]) == LEVEL_FAIR


def test_percentiles_of_records_skip_frames_without_the_value() -> None:
    records = [
        FrameRecord("a", "a", summary=FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, metric_values={"topology_risk": 0.0})),
        FrameRecord("b", "b", summary=FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, metric_values={"topology_risk": 80.0})),
        FrameRecord("c", "c", summary=FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, metric_values={})),
    ]
    percentiles = compute_metric_percentiles(records, "topology_risk")
    assert percentiles == {"a": 100.0, "b": 0.0}


def test_relative_view_of_equal_values_is_the_absolute_one() -> None:
    assert relative_goodness({"a": 0.0, "b": 0.0}, "polygon_bce_score") == {"a": 0.0, "b": 0.0}
    assert relative_goodness({"a": 0.0, "b": 0.0}, "topology_risk") == {"a": 1.0, "b": 1.0}
    stretched = relative_goodness({"a": 10.0, "b": 30.0}, "topology_risk")
    assert stretched == {"a": 1.0, "b": 0.0}


def test_a_frame_without_the_value_is_not_scored(tmp_path: Path) -> None:
    """No photo: the photo check has no value; the frame used to count it as zero — a perfect frame."""

    cv2 = pytest.importorskip("cv2")
    from karakal.core.analytics import compute_build_result_analytics
    from karakal.core.domain import BuildOptions, BuildResult, GeometryMode
    from karakal.core.synthetic_validation import ModelError, ModelPlan, make_validation_frame

    records = []
    for index in range(2):
        frame = make_validation_frame(seed=index, models=(ModelPlan("a", (ModelError("open"),)), ModelPlan("b")))
        paths = {}
        for model in ("a", "b"):
            path = tmp_path / model / f"F_{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame.masks[model])
            paths[model] = str(path)
        records.append(FrameRecord(f"F_{index}", f"F_{index}", model_mask_paths=paths))
    specs = tuple(ModelSpec(model, model, tmp_path / model) for model in ("a", "b"))
    run = BuildResult(records=tuple(records), model_specs=specs, options=BuildOptions(geometry_mode=GeometryMode.MASK, cache_enabled=False))
    result = compute_build_result_analytics(run, metric_key="photo_mismatch_risk")
    assert all(not record.score_ready and record.score_percentile is None for record in result.records)
    scored = compute_build_result_analytics(run, metric_key="topology_risk")
    assert all(record.score_ready for record in scored.records)
