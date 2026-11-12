"""Topology events between model outputs, on synthetic frames with known answers."""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from karakal.core.analytics import compute_build_result_analytics
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, GeometryMode, ModelSpec
from karakal.core.findings import DIAG_MODEL_ERROR, DIAG_UNKNOWN, frame_findings, make_finding
from karakal.core.synthetic_chip import LayerStyle
from karakal.core.synthetic_scorecard import Scorecard, found_from_findings
from karakal.core.synthetic_validation import (
    BENIGN_KINDS,
    CONDUCTOR_ERROR_CODES,
    DATA_DEFECT_KINDS,
    POINT_ERROR_CODES,
    ModelError,
    ModelPlan,
    make_case,
    make_validation_frame,
)
from karakal.core.topology_events import (
    CONSENSUS_ID,
    DETECTOR_TOPOLOGY,
    TOPOLOGY_EVENTS_METRIC,
    TOPOLOGY_RISK_METRIC,
    TOPOLOGY_VERSION,
    compare_to_reference,
    topology_findings,
    topology_metrics,
)

CONDUCTOR_LAYERS = {
    "manhattan": LayerStyle(),
    "diagonals": LayerStyle(diagonals=True),
    "mixed_widths": LayerStyle(mixed_widths=True),
}
VIAS = LayerStyle(vias=True)
SEEDS = range(100, 106)
# Cases whose masks are right: a check of the masks alone must find nothing there.
QUIET_CASES = ("clean", *DATA_DEFECT_KINDS, "misaligned", "uncertain_flat")


def _masks(frame) -> dict[str, np.ndarray]:
    return {model_id: mask > 0 for model_id, mask in frame.masks.items()}


def _card(layer: LayerStyle, cases) -> Scorecard:
    card = Scorecard()
    for case in cases:
        for seed in SEEDS:
            frame = make_case(case, seed, layer)
            found = topology_findings(_masks(frame), point=layer.vias)
            card.add_frame(f"{case}{seed}", frame, found_from_findings(found, frame.shape, model_id="a"), model_id="a")
    return card


@pytest.mark.parametrize("style", sorted(CONDUCTOR_LAYERS))
def test_every_conductor_error_is_found_where_it_is(style: str) -> None:
    card = _card(CONDUCTOR_LAYERS[style], (*CONDUCTOR_ERROR_CODES, "edge_shift"))
    for kind in (*CONDUCTOR_ERROR_CODES, "edge_shift"):
        assert card.recall(kind) >= 0.9, (style, kind, card.to_markdown())
    assert card.false_findings == [], card.false_findings[:5]


def test_every_point_error_is_found_where_it_is() -> None:
    card = _card(VIAS, (*POINT_ERROR_CODES, "edge_shift"))
    for kind in (*POINT_ERROR_CODES, "edge_shift"):
        assert card.recall(kind) >= 0.9, (kind, card.to_markdown())
    assert card.false_findings == [], card.false_findings[:5]


@pytest.mark.parametrize("layer_name", [*sorted(CONDUCTOR_LAYERS), "vias"])
def test_right_masks_and_harmless_jitter_give_no_findings(layer_name: str) -> None:
    layer = VIAS if layer_name == "vias" else CONDUCTOR_LAYERS[layer_name]
    jitter = "point_jitter" if layer.vias else "boundary_jitter"
    for case in (*QUIET_CASES, jitter):
        for seed in SEEDS:
            frame = make_case(case, seed, layer)
            assert topology_findings(_masks(frame), point=layer.vias) == (), (layer_name, case, seed)


def test_a_mask_against_itself_has_nothing() -> None:
    frame = make_case("open", 0)
    mask = frame.masks["a"] > 0
    assert compare_to_reference(mask, mask, model_id="a", reference_id="a") == ()


def test_two_models_see_one_disagreement_from_both_sides() -> None:
    frame = make_case("open", 1)
    found = topology_findings(_masks(frame))
    by_model = {model_id: {item.kind for item in found if item.model_id == model_id} for model_id in ("a", "b")}
    assert by_model == {"a": {"open"}, "b": {"bridge"}}
    assert all(item.diagnosis == DIAG_UNKNOWN and item.detector == DETECTOR_TOPOLOGY for item in found)
    assert all(item.detector_version == TOPOLOGY_VERSION for item in found)
    assert {item.evidence_value("reference") for item in found} == {"a", "b"}
    # Swapping the models swaps the names, nothing else.
    swapped = topology_findings({"a": frame.masks["b"] > 0, "b": frame.masks["a"] > 0})
    assert sorted((item.kind, item.box) for item in swapped) == sorted((item.kind, item.box) for item in found)


def test_missing_in_one_is_extra_in_the_other() -> None:
    frame = make_case("missing_conductor", 2)
    found = topology_findings(_masks(frame))
    assert {(item.model_id, item.kind) for item in found} == {("a", "missing_object"), ("b", "extra_object")}


def test_more_errors_never_lower_the_risk() -> None:
    one = make_validation_frame(seed=4, models=(ModelPlan("a", (ModelError("open"),)), ModelPlan("b")))
    two = make_validation_frame(
        seed=4, models=(ModelPlan("a", (ModelError("open"), ModelError("bridge"), ModelError("extra_blob"))), ModelPlan("b"))
    )
    first = topology_metrics(topology_findings(_masks(one)), 2)
    second = topology_metrics(topology_findings(_masks(two)), 2)
    assert first[TOPOLOGY_EVENTS_METRIC] == 1.0
    assert second[TOPOLOGY_EVENTS_METRIC] == 3.0
    assert 0.0 < first[TOPOLOGY_RISK_METRIC] < second[TOPOLOGY_RISK_METRIC] < 100.0
    assert topology_metrics((), 2) == {TOPOLOGY_RISK_METRIC: 0.0, TOPOLOGY_EVENTS_METRIC: 0.0}


def test_three_models_blame_the_one_against_the_majority() -> None:
    plans = (ModelPlan("a", (ModelError("open"),)), ModelPlan("b"), ModelPlan("c"))
    frame = make_validation_frame(seed=3, models=plans)
    found = topology_findings(_masks(frame))
    assert {(item.model_id, item.kind) for item in found} == {("a", "open")}
    (item,) = found
    assert item.diagnosis == DIAG_MODEL_ERROR and item.evidence_value("reference") == CONSENSUS_ID
    (planted,) = frame.items
    assert found_from_findings(found, frame.shape)[0].box is not None
    card = Scorecard()
    card.add_frame("F", frame, found_from_findings(found, frame.shape, model_id="a"), model_id="a")
    assert card.recall("open") == 1.0 and planted.kind == "open"


@pytest.mark.parametrize("shift", [-2, -1, 1, 2])
def test_frame_wide_edge_shift_is_measured(shift: int) -> None:
    frame = make_validation_frame(seed=5, models=(ModelPlan("a", (ModelError("edge_shift", shift=shift),)), ModelPlan("b")))
    found = [item for item in topology_findings(_masks(frame)) if item.model_id == "a" and item.box is None]
    (item,) = found
    assert item.kind == "boundary_shift"
    assert float(item.evidence_value("shift_px")) == pytest.approx(shift, abs=0.35)


def test_shifted_point_reports_how_far() -> None:
    frame = make_validation_frame(seed=2, layer=VIAS, models=(ModelPlan("a", (ModelError("shifted_point", shift=12),)), ModelPlan("b")))
    (item,) = [item for item in topology_findings(_masks(frame), point=True) if item.model_id == "a"]
    assert item.kind == "shifted_point"
    assert float(item.evidence_value("shift_px")) == pytest.approx(12.0, abs=1.0)


def test_masks_of_other_shapes_are_left_out() -> None:
    frame = make_case("open", 0)
    assert topology_findings({"a": frame.masks["a"] > 0, "b": np.zeros((10, 10), bool)}) == ()
    assert topology_findings({"a": frame.masks["a"] > 0}) == ()
    with pytest.raises(ValueError):
        compare_to_reference(frame.masks["a"] > 0, np.zeros((10, 10), bool), model_id="a", reference_id="b")


def test_metrics_count_two_models_once() -> None:
    finding = make_finding("open", DETECTOR_TOPOLOGY, 0.9)
    other = make_finding("blur", "heuristic_check", 1.0)
    assert topology_metrics((finding, finding, other), 2)[TOPOLOGY_EVENTS_METRIC] == 1.0
    assert topology_metrics((finding, finding, other), 3)[TOPOLOGY_EVENTS_METRIC] == 2.0


# --- Through the analytics ------------------------------------------------------------------------


def _write(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return str(path)


def test_analytics_find_a_small_open_on_a_scaled_down_frame(tmp_path: Path) -> None:
    """The analysis works on 512 px of a 2048 px frame; the three-pixel gap is read at full size."""

    layer = LayerStyle(size=2048)
    frames = {"F_00": make_case("open", 7, layer), "F_01": make_case("clean", 8, layer)}
    records = tuple(
        FrameRecord(key, key, model_mask_paths={m: _write(tmp_path / m / f"{key}.png", frame.masks[m]) for m in ("a", "b")})
        for key, frame in frames.items()
    )
    source = BuildResult(
        records=records,
        model_specs=(ModelSpec("a", "a", tmp_path / "a"), ModelSpec("b", "b", tmp_path / "b")),
        options=BuildOptions(cache_enabled=False, analysis_max_side=512, geometry_mode=GeometryMode.MASK),
    )
    result = compute_build_result_analytics(source, metric_key=TOPOLOGY_RISK_METRIC)
    assert result.selected_metric_key == TOPOLOGY_RISK_METRIC
    by_key = {record.key: record for record in result.records}
    broken, clean = by_key["F_00"].summary, by_key["F_01"].summary
    assert broken.metric_values[TOPOLOGY_EVENTS_METRIC] == 1.0
    assert broken.metric_values[TOPOLOGY_RISK_METRIC] > 0.0
    assert clean.metric_values[TOPOLOGY_RISK_METRIC] == 0.0
    assert {item.kind for item in frame_findings(broken) if item.model_id == "a"} == {"open"}
    # The pixel overlap the analysis sees hardly notices it.
    assert broken.metric_values["iou"] > 0.99
    # Lower is better: the broken frame ranks worst.
    assert by_key["F_00"].score < by_key["F_01"].score


def test_point_layers_are_compared_as_points(tmp_path: Path) -> None:
    frames = {f"F_{index:02d}": make_case(case, 20 + index, VIAS) for index, case in enumerate(("merged_points", "clean"))}
    records = tuple(
        FrameRecord(key, key, model_mask_paths={m: _write(tmp_path / m / f"{key}.png", frame.masks[m]) for m in ("a", "b")})
        for key, frame in frames.items()
    )
    source = BuildResult(
        records=records,
        model_specs=(ModelSpec("a", "a", tmp_path / "a"), ModelSpec("b", "b", tmp_path / "b")),
        options=BuildOptions(cache_enabled=False, geometry_mode=GeometryMode.POINT),
    )
    result = compute_build_result_analytics(source, metric_key=TOPOLOGY_RISK_METRIC)
    kinds = {item.kind for item in frame_findings(result.records[0].summary) if item.model_id == "a"}
    assert kinds == {"merged_points"}
    assert frame_findings(result.records[1].summary) == ()
