"""One format for what validation checks find: the format, its adapters, old results, the bench."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from karakal.core.analytics import compute_build_result_analytics
from karakal.core.attention_issues import AttentionIssue
from karakal.core.domain import BuildOptions, BuildResult, FrameAnalysisSummary, FrameRecord, ModelSpec
from karakal.core.findings import (
    DETECTOR_ATTENTION,
    DETECTOR_HEURISTIC_CHECK,
    DETECTOR_SINGLE_RESULT_RISK,
    DIAG_BAD_DATA,
    DIAG_MODEL_ERROR,
    DIAG_UNCERTAIN,
    DIAG_UNKNOWN,
    FINDINGS_SCHEMA_VERSION,
    Finding,
    findings_from_attention_issues,
    findings_from_heuristic_check,
    findings_from_payload,
    findings_from_risk_reasons,
    findings_from_single_result_risk,
    findings_to_payload,
    frame_findings,
    make_finding,
    severity_label,
    sort_findings,
    summarize_findings,
)
from karakal.core.heuristic_check import HEURISTIC_CHECK_VERSION, compute_heuristic_check
from karakal.core.single_result_risk import SINGLE_RESULT_RISK_VERSION, RiskReason
from karakal.core.synthetic_scorecard import found_from_findings
from karakal.core.synthetic_validation import make_case


# --- The format --------------------------------------------------------------------------------------


def test_make_finding_cleans_its_values() -> None:
    finding = make_finding("open", "topology", 1.7, (0.1, 0.2, 0.3, 0.4), diagnosis="nonsense", evidence={"b": 2, "a": "x"})
    assert finding.severity == 1.0
    assert finding.box == (0.1, 0.2, 0.3, 0.4)
    assert finding.diagnosis == DIAG_UNKNOWN
    assert finding.evidence == (("a", "x"), ("b", 2.0))
    assert finding.evidence_value("b") == 2.0
    assert make_finding("x", "d", float("nan")).severity == 0.0
    assert make_finding("x", "d", 0.5, (0.1, 0.1, -1.0, 0.1)).box is None
    assert make_finding("x", "d", 0.5, "not a box").whole_frame


@pytest.mark.parametrize(("value", "label"), [(0.1, "low"), (0.6, "medium"), (0.8, "high"), (0.95, "critical")])
def test_severity_words_match_the_risk_reasons(value: float, label: str) -> None:
    assert severity_label(value) == label
    assert make_finding("x", "d", value).severity_label == label


def test_pixel_box_scales_to_the_frame() -> None:
    finding = make_finding("x", "d", 0.5, (0.25, 0.5, 0.1, 0.2))
    assert finding.pixel_box((200, 400)) == (100.0, 100.0, 40.0, 40.0)
    assert make_finding("x", "d", 0.5).pixel_box((10, 10)) is None


def test_findings_survive_json_and_back() -> None:
    findings = (
        make_finding("open", "topology", 0.8, (0.1, 0.2, 0.05, 0.05), diagnosis=DIAG_MODEL_ERROR, model_id="a",
                     detector_version="topology.v1", evidence={"gap_px": 4}, detail="2"),
        make_finding("blur", "heuristic_check", 1.0, None, diagnosis=DIAG_BAD_DATA),
        Finding("merge", "attention", 0.4, (0.0, 0.0, 0.5, 0.5), outline=((0.0, 0.0), (0.5, 0.0), (0.5, 0.5))),
    )
    text = json.dumps(findings_to_payload(findings))
    payload = json.loads(text)
    assert payload["schema_version"] == FINDINGS_SCHEMA_VERSION
    assert findings_from_payload(payload) == findings
    # A bare list reads too.
    assert findings_from_payload(payload["findings"]) == findings


def test_unreadable_findings_are_skipped() -> None:
    payload = [
        None,
        "text",
        {"kind": "", "detector": "d"},
        {"kind": "k"},
        {"kind": "k", "detector": "d", "severity": "bad", "box": [1, 2], "diagnosis": "?", "outline": [[1], "x", [0.1, 0.2]]},
    ]
    (finding,) = findings_from_payload(payload)
    assert finding.severity == 0.0 and finding.box is None and finding.diagnosis == DIAG_UNKNOWN
    assert finding.outline == ((0.1, 0.2),)
    assert findings_from_payload(42) == ()


def test_sort_puts_the_worst_first_and_counts_kinds() -> None:
    findings = [make_finding("b", "d", 0.2), make_finding("a", "d", 0.9), make_finding("b", "d", 0.5, (0.1, 0.1, 0.1, 0.1))]
    assert [item.severity for item in sort_findings(findings)] == [0.9, 0.5, 0.2]
    assert summarize_findings(findings) == {"a": 1, "b": 2}


# --- Adapters ----------------------------------------------------------------------------------------


def test_risk_reasons_become_findings() -> None:
    reasons = (
        RiskReason("tiny_components", "medium", 0.6, (0.1, 0.1, 0.2, 0.2), "3"),
        RiskReason("unsupported_boundary", "high", 0.8, (0.5, 0.5, 0.1, 0.1)),
        RiskReason("batch_outlier.area_fraction", "low", 0.4),
        RiskReason("insufficient_baseline", "low", 0.0, detail="10"),
    )
    found = findings_from_risk_reasons(
        reasons,
        detector="risk",
        model_id="a",
        diagnoses={"tiny_components": DIAG_MODEL_ERROR, "batch_outlier": DIAG_BAD_DATA},
    )
    assert [item.kind for item in found] == ["tiny_components", "unsupported_boundary", "batch_outlier.area_fraction"]
    tiny, boundary, batch = found
    assert tiny.severity == 0.6 and tiny.box == (0.1, 0.1, 0.2, 0.2) and tiny.detail == "3"
    assert tiny.diagnosis == DIAG_MODEL_ERROR and tiny.model_id == "a"
    assert tiny.severity_label == "medium" and tiny.evidence == ()
    assert boundary.diagnosis == DIAG_UNKNOWN
    assert batch.box is None and batch.diagnosis == DIAG_BAD_DATA


def test_attention_issues_become_findings_from_objects_and_dicts() -> None:
    issue = AttentionIssue("i1", "break", "high", 0.7, (0.1, 0.1, 0.2, 0.2), "attention.issue.break", model_id="a")
    other = AttentionIssue("i2", "original_mismatch", "low", 0.3, (0.6, 0.6, 0.1, 0.1), "x", model_id="b").to_dict()
    found = findings_from_attention_issues({"a": (issue,), "b": [other, {"broken": True}]})
    assert [(item.kind, item.model_id, item.diagnosis) for item in found] == [
        ("break", "a", DIAG_UNCERTAIN),
        ("original_mismatch", "b", DIAG_UNKNOWN),
    ]
    assert all(item.detector == DETECTOR_ATTENTION for item in found)
    assert findings_from_attention_issues(None) == ()


# --- On real runs over synthetic frames ---------------------------------------------------------------


def _write(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return str(path)


CASES = ("clean", "open", "extra_blob", "residue", "particles", "clean", "missing_conductor", "edge_shift")


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("findings_run")
    frames = {}
    for index, case in enumerate(CASES):
        frame = make_case(case, index)
        key = f"F_{index:02d}"
        frames[key] = frame
        _write(root / "photos" / f"{key}.png", frame.photo)
        for model_id in ("a", "b"):
            _write(root / model_id / f"{key}.png", frame.masks[model_id])
            _write(root / model_id / f"{key}_confidence.png", frame.confidences[model_id])
    return root, frames


def _records(root: Path, frames: dict, model_ids: tuple[str, ...]) -> tuple[FrameRecord, ...]:
    return tuple(
        FrameRecord(
            key,
            key,
            original_path=str(root / "photos" / f"{key}.png"),
            model_mask_paths={m: str(root / m / f"{key}.png") for m in model_ids},
            model_prob_paths={m: str(root / m / f"{key}_confidence.png") for m in model_ids},
        )
        for key in sorted(frames)
    )


def test_single_result_risk_summaries_carry_their_findings(synthetic_run) -> None:
    root, frames = synthetic_run
    source = BuildResult(
        records=_records(root, frames, ("a",)),
        model_specs=(ModelSpec("a", "a", root / "a", prob_folder=root / "a"),),
        options=BuildOptions(cache_enabled=False, max_workers=2),
    )
    result = compute_build_result_analytics(source, metric_key="single_result_risk_score")
    for record in result.records:
        summary = record.summary
        risk = summary.single_result_risk
        expected = findings_from_single_result_risk(risk, model_id="a")
        assert summary.findings == expected
        codes = [reason.code for reason in risk.reasons if reason.code != "insufficient_baseline"]
        assert sorted(item.kind for item in summary.findings) == sorted(codes)
        for item in summary.findings:
            assert item.detector == DETECTOR_SINGLE_RESULT_RISK
            assert item.detector_version == SINGLE_RESULT_RISK_VERSION
            assert item.model_id == "a"


def test_heuristic_check_findings_list_every_region_as_bad_data(synthetic_run) -> None:
    root, frames = synthetic_run
    source = BuildResult(
        records=tuple(FrameRecord(key, key, model_mask_paths={"photos": str(root / "photos" / f"{key}.png")}) for key in sorted(frames)),
        model_specs=(ModelSpec("photos", "Photos", root / "photos"),),
        options=BuildOptions(cache_enabled=False),
    )
    result = compute_heuristic_check(source)
    saw_regions = False
    for record in result.records:
        check = record.summary.heuristic_check
        findings = record.summary.findings
        assert findings == findings_from_heuristic_check(check)
        assert all(item.detector == DETECTOR_HEURISTIC_CHECK and item.diagnosis == DIAG_BAD_DATA for item in findings)
        assert all(item.detector_version == HEURISTIC_CHECK_VERSION and item.model_id == "" for item in findings)
        boxed = [item for item in findings if item.box is not None]
        assert len(boxed) == len(check.regions)
        saw_regions = saw_regions or bool(check.regions)
        # Reasons of the whole frame are not lost.
        whole = {reason.code for reason in check.reasons if reason.bbox is None}
        assert whole <= {item.kind for item in findings if item.box is None}
    assert saw_regions


def test_model_comparison_findings_come_from_attention_issues(synthetic_run) -> None:
    root, frames = synthetic_run
    model_ids = ("a", "b")
    source = BuildResult(
        records=_records(root, frames, model_ids),
        model_specs=tuple(ModelSpec(m, m, root / m, prob_folder=root / m) for m in model_ids),
        options=BuildOptions(cache_enabled=False, max_workers=2),
    )
    result = compute_build_result_analytics(source, metric_key="overall_frame_score", include_attention_issues=True)
    total = 0
    for record in result.records:
        summary = record.summary
        attention = findings_from_attention_issues(summary.attention_issues_by_model)
        assert set(attention) <= set(summary.findings)
        assert all(item.detector in {DETECTOR_ATTENTION, "topology"} for item in summary.findings)
        total += len(attention)
    assert total > 0
    # Without attention issues a model comparison lists its topology events only.
    plain = compute_build_result_analytics(source, metric_key="overall_frame_score")
    assert all(item.detector == "topology" for record in plain.records for item in record.summary.findings)


def test_summaries_saved_before_findings_get_them_from_their_checks(synthetic_run) -> None:
    root, frames = synthetic_run
    source = BuildResult(
        records=_records(root, frames, ("a",)),
        model_specs=(ModelSpec("a", "a", root / "a"),),
        options=BuildOptions(cache_enabled=False),
    )
    record = next(
        record
        for record in compute_build_result_analytics(source, metric_key="single_result_risk_score").records
        if record.summary.findings
    )
    old = pickle.loads(pickle.dumps(record.summary))
    del old.findings  # what a summary pickled before the slot existed looks like
    assert getattr(old, "findings", None) is None
    derived = frame_findings(old)
    # The model is not known from an old risk summary; everything else is.
    assert [(item.kind, item.box, item.severity) for item in derived] == [
        (item.kind, item.box, item.severity) for item in record.summary.findings
    ]
    assert frame_findings(record.summary) == record.summary.findings
    assert frame_findings(None) == ()
    assert frame_findings(FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0)) == ()


def test_findings_feed_the_scorecard_in_frame_pixels() -> None:
    findings = (
        make_finding("open", "topology", 0.9, (0.5, 0.25, 0.1, 0.1), diagnosis=DIAG_MODEL_ERROR, model_id="a"),
        make_finding("open", "topology", 0.9, (0.5, 0.25, 0.1, 0.1), model_id="b"),
        make_finding("residue", "heuristic_check", 0.5, None, diagnosis=DIAG_UNKNOWN),
    )
    items = found_from_findings(findings, (200, 400), model_id="a")
    assert [(item.code, item.box, item.diagnosis, item.model_id) for item in items] == [
        ("open", (200.0, 50.0, 40.0, 20.0), DIAG_MODEL_ERROR, "a"),
        ("residue", None, None, None),
    ]
    assert len(found_from_findings(findings, (200, 400))) == 3


def test_photo_items_are_asked_of_a_photo_check() -> None:
    from karakal.core.synthetic_validation import ModelError, ModelPlan, make_validation_frame

    frame = make_validation_frame(seed=0, models=(ModelPlan("a", (ModelError("open"),)),), data_defect="residue")
    assert {item.kind for item in frame.items_for("")} == {"residue"}
    assert {item.kind for item in frame.items_for("a")} == {"open", "residue"}
    assert not math.isnan(len(frame.items_for(None)))
