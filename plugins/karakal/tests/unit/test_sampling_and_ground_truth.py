"""Picking a labelling sample, exporting it with a manifest, scoring it against ground truth."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from karakal.core.domain import (
    BuildResult,
    FrameAnalysisSummary,
    FrameIdentity,
    FrameRecord,
    ModelSpec,
)
from karakal.core.findings import DIAG_MODEL_ERROR, make_finding
from karakal.core.ground_truth import evaluate, evaluate_manifest, find_ground_truth
from karakal.core.sample_export import export_sample, read_manifest
from karakal.core.sampling import (
    STRATEGY_DIVERSE,
    STRATEGY_PER_KIND,
    STRATEGY_RANDOM,
    STRATEGY_WORST,
    SamplePlan,
    SampleRule,
    append_exclusions,
    build_sample,
    default_plan,
    read_exclusions,
)
from karakal.core.synthetic_validation import make_case
from karakal.core.topology_events import topology_findings

KINDS = ("open", "bridge", "missing_object", "neck")


def _run(count: int = 100, *, with_places: bool = True) -> BuildResult:
    rng = np.random.default_rng(0)
    records = []
    for index in range(count):
        kind = KINDS[index % len(KINDS)] if index % 3 == 0 else None
        findings = (make_finding(kind, "topology", 0.9, (0.1, 0.1, 0.05, 0.05), diagnosis=DIAG_MODEL_ERROR, model_id="a"),) if kind else ()
        summary = FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, metric_values={"topology_risk": float(rng.uniform(0, 100))}, findings=findings)
        identity = FrameIdentity(index, tile_x=index % 10, tile_y=index // 10) if with_places else None
        records.append(FrameRecord(f"F_{index:03d}", f"F_{index:03d}", identity=identity, summary=summary))
    return BuildResult(records=tuple(records), model_specs=(ModelSpec("a", "a", Path(".")),))


def test_default_plan_parts_add_up() -> None:
    plan = default_plan("topology_risk", total=40)
    assert sum(rule.count for rule in plan.rules) == 40
    assert [rule.strategy for rule in plan.rules] == [STRATEGY_WORST, STRATEGY_PER_KIND, STRATEGY_DIVERSE, STRATEGY_RANDOM]
    assert all(rule.count > 0 for rule in plan.rules)


def test_sample_is_the_same_for_one_seed_and_never_repeats_a_frame() -> None:
    run = _run()
    first = build_sample(run, default_plan("topology_risk", total=40, seed=5))
    again = build_sample(run, default_plan("topology_risk", total=40, seed=5))
    other = build_sample(run, default_plan("topology_risk", total=40, seed=6))
    assert [frame.key for frame in first] == [frame.key for frame in again]
    assert len({frame.key for frame in first}) == len(first) == 40
    assert {frame.key for frame in first if frame.strategy == STRATEGY_RANDOM} != {
        frame.key for frame in other if frame.strategy == STRATEGY_RANDOM
    }


def test_worst_takes_the_highest_risk() -> None:
    run = _run()
    sample = build_sample(run, SamplePlan((SampleRule(STRATEGY_WORST, 5, metric_key="topology_risk"),)))
    values = sorted((record.summary.metric_values["topology_risk"] for record in run.records), reverse=True)[:5]
    assert [frame.metric_value for frame in sample] == values


def test_per_kind_gives_every_kind_its_share() -> None:
    sample = build_sample(_run(), SamplePlan((SampleRule(STRATEGY_PER_KIND, 8),)))
    kinds = [frame.reason.split()[1] for frame in sample]
    assert sorted(kinds) == sorted(KINDS * 2)
    only = build_sample(_run(), SamplePlan((SampleRule(STRATEGY_PER_KIND, 5, kinds=("bridge",)),)))
    assert all(frame.reason.startswith("has bridge") for frame in only) and len(only) == 5
    # No finding of that diagnosis: the rule's share goes to chance, the sample keeps its size.
    none = build_sample(_run(), SamplePlan((SampleRule(STRATEGY_PER_KIND, 5, diagnosis="bad_data"),)))
    assert len(none) == 5 and all(frame.strategy == STRATEGY_RANDOM and "filling" in frame.reason for frame in none)


def test_diverse_frames_lie_far_apart_on_the_matrix() -> None:
    run = _run()
    sample = build_sample(run, SamplePlan((SampleRule(STRATEGY_DIVERSE, 4),)))
    places = {record.key: (record.identity.tile_x, record.identity.tile_y) for record in run.records}
    points = np.asarray([places[frame.key] for frame in sample], dtype=float)
    nearest = min(np.hypot(*(points[i] - points[j])) for i in range(4) for j in range(i + 1, 4))
    assert nearest >= 6.0
    # Without matrix places the run order and the metric spread them.
    loose = build_sample(_run(with_places=False), SamplePlan((SampleRule(STRATEGY_DIVERSE, 4, metric_key="topology_risk"),)))
    assert len({frame.key for frame in loose}) == 4


def test_random_part_is_uniform() -> None:
    run = _run(50)
    hits = np.zeros(50)
    for seed in range(400):
        for frame in build_sample(run, SamplePlan((SampleRule(STRATEGY_RANDOM, 5),), seed=seed)):
            hits[int(frame.key[2:])] += 1
    expected = 400 * 5 / 50
    assert hits.min() > 0.5 * expected and hits.max() < 1.5 * expected


def test_excluded_frames_are_never_taken(tmp_path: Path) -> None:
    registry = tmp_path / "labelled.txt"
    run = _run()
    first = build_sample(run, default_plan("topology_risk", total=30, exclude=read_exclusions(registry)))
    append_exclusions(registry, (frame.key for frame in first))
    append_exclusions(registry, (frame.key for frame in first))  # twice: no duplicates
    assert len(registry.read_text(encoding="utf-8").splitlines()) == 30
    second = build_sample(run, default_plan("topology_risk", total=30, exclude=read_exclusions(registry)))
    assert not ({frame.key for frame in first} & {frame.key for frame in second})
    assert read_exclusions(tmp_path / "missing.txt") == frozenset()
    with pytest.raises(ValueError):
        build_sample(run, SamplePlan((SampleRule("nonsense", 1),)))


# --- Export and ground truth on synthetic frames ---------------------------------------------------


def _write(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return str(path)


@pytest.fixture()
def synthetic(tmp_path: Path):
    cases = ("open", "bridge", "missing_conductor", "clean", "boundary_jitter", "residue")
    frames, records = {}, []
    for index, case in enumerate(cases):
        frame = make_case(case, 100 + index)
        key = f"F_{index:02d}"
        frames[key] = frame
        found = topology_findings({m: frame.masks[m] > 0 for m in ("a", "b")})
        summary = FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, metric_values={"topology_risk": float(len(found))}, findings=found)
        records.append(
            FrameRecord(
                key,
                key,
                original_path=_write(tmp_path / "photos" / f"{key}.png", frame.photo),
                model_mask_paths={m: _write(tmp_path / m / f"{key}.png", frame.masks[m]) for m in ("a", "b")},
                summary=summary,
            )
        )
        _write(tmp_path / "truth" / f"{key}.png", (frame.truth * 255).astype(np.uint8))
    run = BuildResult(records=tuple(records), model_specs=(ModelSpec("a", "a", tmp_path / "a"), ModelSpec("b", "b", tmp_path / "b")))
    return tmp_path, run, frames


def test_export_writes_files_and_a_manifest(synthetic) -> None:
    root, run, _frames = synthetic
    plan = SamplePlan((SampleRule(STRATEGY_WORST, 3, metric_key="topology_risk"), SampleRule(STRATEGY_RANDOM, 2)), seed=1)
    sample = build_sample(run, plan)
    manifest_path = export_sample(run, sample, root / "out", plan=plan, prelabel_model="a")
    manifest = read_manifest(root / "out")
    assert manifest_path.name == "manifest.json" and manifest["prelabel_model"] == "a"
    assert manifest["plan"]["seed"] == 1 and len(manifest["frames"]) == 5
    assert manifest["detectors"] == {"topology": "topology.v1"}
    for row in manifest["frames"]:
        files = row["files"]
        for relative in (files["photo"], files["prelabel"], files["overlay"], *files["masks"].values()):
            assert (root / "out" / relative).is_file()
        assert row["strategy"] in {STRATEGY_WORST, STRATEGY_RANDOM}
    csv_rows = (root / "out" / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_rows) == 6


def test_ground_truth_confirms_the_real_errors(synthetic) -> None:
    root, run, frames = synthetic
    frames_info = {
        record.key: {"masks": record.model_mask_paths, "findings": record.summary.findings, "strategy": "all"}
        for record in run.records
    }
    report = evaluate(frames_info, find_ground_truth(root / "truth", frames_info))
    assert report.missing_truth == []
    real = {(item.frame, item.model_id): {event.kind for event in item.events} for item in report.frames}
    assert real[("F_00", "a")] == {"open"} and real[("F_00", "b")] == set()
    assert real[("F_01", "a")] == {"bridge"}
    assert real[("F_02", "a")] == {"missing_object"}
    assert real[("F_03", "a")] == set() and real[("F_04", "a")] == set()
    # Model a's findings are its real errors; b's mirrored ones are not b's.
    assert report.recall["open"].share == 1.0 and report.recall["bridge"].share == 1.0
    assert report.precision[("topology", "open")].hit >= 1
    ious = {(item.frame, item.model_id): item.iou for item in report.frames}
    assert ious[("F_03", "a")] == 1.0 and ious[("F_00", "a")] < 1.0
    assert report.error_rate["all"].total == 6 and report.error_rate["all"].hit == 3
    text = report.to_markdown()
    assert "precision" in text and "random part" in text


def test_frames_without_truth_are_listed(synthetic) -> None:
    root, run, _frames = synthetic
    (root / "truth" / "F_05.png").unlink()
    frames_info = {record.key: {"masks": record.model_mask_paths, "findings": (), "strategy": "all"} for record in run.records}
    report = evaluate(frames_info, find_ground_truth(root / "truth", frames_info))
    assert report.missing_truth == ["F_05"]


def _script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"karakal_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_end_to_end_synthetic_truth_through_the_scripts(tmp_path: Path) -> None:
    """Synthetic truth stands in for ground truth: the whole cycle runs before real truth exists."""

    folder, frames = _script("validation_synth_set").write_style(
        tmp_path, "manhattan", per_case=1, clean=4, model_count=2, behaviour="calibrated",
        encoding="probability", shared=False, reaction="ignores",
    )
    registry = tmp_path / "labelled.txt"
    code = _script("build_sample").main(
        [str(tmp_path / "sample"), "--photos", str(folder / "photos"), "--model", f"a={folder / 'model_a'}",
         "--model", f"b={folder / 'model_b'}", "--total", "16", "--seed", "2", "--exclude", str(registry)]
    )
    assert code == 0
    manifest = json.loads((tmp_path / "sample" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["frames"]) == 16 and len(read_exclusions(registry)) == 16
    assert _script("evaluate_with_ground_truth").main([str(tmp_path / "sample"), str(folder / "truth")]) == 0
    report = json.loads((tmp_path / "sample" / "ground_truth_report.json").read_text(encoding="utf-8"))
    assert report["missing_truth"] == []
    # Every frame of the sample whose model a carries a planted error has a real event of a.
    real_a = {row["frame"]: row["events"] for row in report["frames"] if row["model"] == "a"}
    for row in manifest["frames"]:
        frame = frames[Path(row["frame"]).stem]  # keys read from folders keep the file's extension
        planted = [item for item in frame.items_for("a") if item.models == ("a",) and item.is_positive]
        if planted:
            assert real_a[row["frame"]], (row["frame"], planted)
        elif frame.case == "clean":
            assert real_a[row["frame"]] == []
    # The checks found every real event (the topology detector agrees with the truth here).
    assert all(item["recall"] == 1.0 for item in report["recall"])
    assert evaluate_manifest(tmp_path / "sample", folder / "truth").frames
