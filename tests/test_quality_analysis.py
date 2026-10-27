"""Quality rules consume real Karakal metrics imported into AnalysisRun stores."""

from dataclasses import replace
from contextlib import closing
import platform
import shutil
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from kraken_core.analysis_protocol import AnalysisArtifactInput, AnalysisFrameInput, AnalysisSourceRole
from kraken_core.analysis_run_protocol import AnalysisExpression, AnalysisRecipe, AnalysisRuntimeIdentity, AnalysisSourceBinding
from kraken_manager.application.analysis_runs import AnalysisRunCoordinator
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.quality import QualityState
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
from karakal.worker import execute
from tests.test_quality_workflow import quality as quality


RULE = {"metric_key": "iou", "value_field": "raw_value", "comparison": "lt", "threshold": 0.9, "target_binding": "A"}


def real_inputs(q, root):
    versions = []
    for name, position in (("target", (1, 1)), ("reference", (6, 6)), ("corrected", (6, 6))):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[position] = 255
        path = root / f"{name}.png"
        assert cv2.imwrite(str(path), mask)
        version = q.service.add_managed_artifact_version(principal=q.owner, project_id=q.project.id,
            series_id=q.source.series_id, source=path,
            parent_version_id=(versions[0].id if name == "corrected" else q.corrected.id), idempotency_key=f"png-{name}")
        versions.append(version)
    return tuple(versions)


def real_run(q, root, store, target, reference, *, imported=True, runtime=None, legacy=False):
    root.mkdir(parents=True)
    artifacts = []
    for key, version in (("A", target), ("B", reference)):
        relative = f"inputs/{key}.png"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(q.service.artifact_source_path(q.project.id, version.id), path)
        artifacts.append(AnalysisArtifactInput(key, AnalysisSourceRole.MODEL_OUTPUT, str(version.series_id),
            str(version.id), relative, "image/png", version.sha256))
    frame = AnalysisFrameInput(str(q.project.frame_id_at(1, 1)), 1, 1, tuple(artifacts))
    run, partitions = AnalysisRunCoordinator.build_contracts(project_id=str(q.project.id), frames=(frame,),
        source_bindings=(AnalysisSourceBinding("A", "target", "1"), AnalysisSourceBinding("B", "reference", "1")),
        recipe=AnalysisRecipe(AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B")),
                              metric_keys=("iou",)),
        runtime=runtime or AnalysisRuntimeIdentity("karakal-headless", "quality-regression", platform.python_version(),
                                                  np.__version__, cv2.__version__, platform.system()))
    if legacy:
        partitions = tuple(replace(partition, require_worker_runtime=False) for partition in partitions)
    store.create_run(run, partitions)
    result = None
    if imported:
        partition = partitions[0]
        partition.write(root / "job.json")
        result = execute(root / "job.json", root / "result.json", root / "progress.json", root)
        if legacy:
            result = replace(result, worker_runtime=None)  # Historical contract fixture, not a measured receipt.
        if isinstance(store, FilesystemAnalysisStore):
            store.import_partition(result, root / result.bundle.relative_path)
        else:
            from kraken_core.analysis_bundle import stream_bundle_records
            with (root / result.bundle.relative_path).open("rb") as stream:
                frames = tuple(stream_bundle_records(stream, result.bundle, expected_frame_ids=run.frame_ids))
            store.import_partition(result, frames)
    return run, result


@pytest.fixture
def analysis(quality, tmp_path):
    q = quality
    target, reference, correction = real_inputs(q, tmp_path)
    store = FilesystemAnalysisStore(q.service.catalog_root, str(q.project.id))
    source, _ = real_run(q, tmp_path / "original-run", store, target, reference)
    checked, _ = real_run(q, tmp_path / "corrected-run", store, correction, reference)
    return SimpleNamespace(q=q, store=store, source=source, checked=checked,
                           target=target, reference=reference, correction=correction, root=tmp_path)


def findings(a, *, rule=None, key="metric-findings"):
    return a.q.service.create_quality_from_analysis(project_id=a.q.project.id, principal=a.q.owner,
        run_id=a.source.run_id, rule=rule or RULE, severity="major", idempotency_key=key)


def corrected_finding(a):
    q = a.q
    issue = findings(a)[0]
    issue = q.command(issue, "assign", {"assignee_id": str(q.performer.id), "note": "Repair target mask"})
    return q.command(issue, "correct", {"evidence_version_ids": [str(a.correction.id)], "note": "Mask repaired"})


def test_real_metric_rule_findings_recheck_and_run_readiness(analysis):
    a, q = analysis, analysis.q
    assert dict(a.store.frame_results(a.source.run_id, "iou")[0])["raw_value"] == 0.0
    assert dict(a.store.frame_results(a.checked.run_id, "iou")[0])["raw_value"] == 1.0
    created = findings(a)
    assert len(created) == 1
    assert findings(a) == created
    assert created[0].analysis_run_id == a.source.run_id
    assert created[0].analysis_rule.threshold == 0.9
    assert created[0].source_version_ids == (str(a.target.id),)
    assert not q.service.quality_analysis_readiness(q.project.id, a.source.run_id)["ready"]
    issue = corrected_finding(a)
    with pytest.raises(DomainValidationError, match="completed stored analysis"):
        q.command(issue, "recheck", {"passed": True, "evidence_version_ids": [str(a.correction.id)], "note": "Spoofed result"},
                    principal=q.reviewer)
    checked = q.service.recheck_quality_from_analysis(project_id=q.project.id, principal=q.reviewer,
        issue_id=issue.id, run_id=a.checked.run_id, expected_revision=issue.revision,
        idempotency_key="real-recheck", note="Independent recheck with the unchanged reference")
    assert checked.state is QualityState.RECHECKED and checked.reviews[-1].passed
    assert checked.reviews[-1].analysis_run_id == a.checked.run_id
    assert checked.reviews[-1].observations == ((str(q.project.frame_id_at(1, 1)), 1.0),)
    accepted = q.command(checked, "accept", {"note": "Measured IoU satisfies the recorded rule"}, principal=q.reviewer)
    assert accepted.state is QualityState.ACCEPTED
    assert q.service.quality_analysis_readiness(q.project.id, a.checked.run_id)["ready"]
    assert not q.service.quality_analysis_readiness(q.project.id, a.source.run_id)["ready"]


def test_recheck_rejects_changed_reference_or_unanalyzed_correction(analysis):
    a, q = analysis, analysis.q
    issue = corrected_finding(a)
    changed, _ = real_run(q, a.root / "changed-reference", a.store, a.correction, a.correction)
    for run, message in ((a.source, "recorded correction"), (changed, "unaffected reference")):
        with pytest.raises(DomainValidationError, match=message):
            q.service.recheck_quality_from_analysis(project_id=q.project.id, principal=q.reviewer,
                issue_id=issue.id, run_id=run.run_id, expected_revision=issue.revision,
                idempotency_key=f"bad-recheck:{run.run_id}", note="Must not pass")
    assert q.service.get_quality_issue(q.project.id, issue.id).state is QualityState.CORRECTED


def test_recheck_rejects_a_different_runtime_or_recipe_identity(analysis, monkeypatch):
    a, q = analysis, analysis.q
    issue = corrected_finding(a)
    # Controlled receipt fault: calculations still use the real Karakal engine.
    # Merely changing Hub's declared runtime is no longer evidence of a different worker.
    actual = a.store.get_run(a.source.run_id).worker_runtime
    runtime = replace(actual, engine_build="sha256:" + "f" * 64)
    with monkeypatch.context() as scoped:
        scoped.setattr("karakal.worker.worker_runtime_identity", lambda *_args: runtime)
        changed, _ = real_run(q, a.root / "different-runtime", a.store, a.correction, a.reference)
    assert changed.runtime == a.source.runtime
    assert a.store.get_run(changed.run_id).worker_runtime == runtime
    with pytest.raises(DomainValidationError, match="same analysis recipe"):
        q.service.recheck_quality_from_analysis(project_id=q.project.id, principal=q.reviewer,
            issue_id=issue.id, run_id=changed.run_id, expected_revision=issue.revision,
            idempotency_key="changed-runtime", note="Incomparable measurements")


def test_legacy_analysis_is_readable_but_cannot_approve_runtime_comparability(analysis):
    a, q = analysis, analysis.q
    old, _ = real_run(q, a.root / "legacy-no-receipt", a.store, a.target, a.reference, legacy=True)
    assert a.store.get_run(old.run_id).worker_runtime is None
    assert len(a.store.frame_results(old.run_id, "iou")) == 1
    with pytest.raises(DomainValidationError, match="measured worker runtime"):
        q.service.create_quality_from_analysis(project_id=q.project.id, principal=q.owner, run_id=old.run_id,
            rule=RULE, severity="major", idempotency_key="legacy-runtime-refused")


def test_changed_host_declaration_does_not_override_matching_measured_worker(analysis):
    a, q = analysis, analysis.q
    issue = corrected_finding(a)
    host_declaration = replace(a.source.runtime, engine_build="another-hub-declaration")
    changed, _ = real_run(q, a.root / "different-host-declaration", a.store, a.correction, a.reference,
                          runtime=host_declaration)
    assert changed.runtime != a.source.runtime
    assert a.store.get_run(changed.run_id).worker_runtime == a.store.get_run(a.source.run_id).worker_runtime
    checked = q.service.recheck_quality_from_analysis(project_id=q.project.id, principal=q.reviewer,
        issue_id=issue.id, run_id=changed.run_id, expected_revision=issue.revision,
        idempotency_key="measured-worker-over-host-declaration", note="Same measured worker, changed client metadata")
    assert checked.state is QualityState.RECHECKED


def test_missing_metric_incomplete_run_and_zero_findings_are_not_silently_passing(analysis):
    a, q = analysis, analysis.q
    with pytest.raises(DomainValidationError, match="stored finite metric"):
        findings(a, rule=RULE | {"metric_key": "not-computed"})
    queued, _ = real_run(q, a.root / "queued-run", a.store, a.target, a.reference, imported=False)
    assert not q.service.quality_analysis_readiness(q.project.id, queued.run_id)["computational_ready"]
    with pytest.raises(ConflictError, match="fully imported"):
        q.service.create_quality_from_analysis(project_id=q.project.id, principal=q.owner, run_id=queued.run_id,
            rule=RULE, severity="major", idempotency_key="incomplete")
    no_findings = RULE | {"comparison": "gt", "threshold": 1.0}
    assert findings(a, rule=no_findings, key="zero-findings") == ()
    assert findings(a, rule=no_findings, key="zero-findings") == ()
    with pytest.raises(ConflictError, match="another analysis"):
        findings(a, key="zero-findings")
    assert q.service.list_quality_issues(q.project.id) == ()


def test_completed_counters_do_not_hide_missing_actual_frame_records(analysis):
    a, q = analysis, analysis.q
    with a.store._connect() as connection:
        connection.execute("DELETE FROM analysis_frame_results WHERE run_id=?", (a.source.run_id,))
    readiness = q.service.quality_analysis_readiness(q.project.id, a.source.run_id)
    assert not readiness["computational_ready"]
    assert "incomplete" in readiness["reason"]
    a.store.rebuild()
    assert q.service.quality_analysis_readiness(q.project.id, a.source.run_id)["computational_ready"]


def test_analysis_cache_survives_metadata_rebuild_and_replays_legacy_cache(analysis):
    a, q = analysis, analysis.q
    findings(a)
    for _ in range(2):
        q.service._projection(q.project.id).destroy_cache()
        assert q.service.get_project(q.project.id) is not None
        assert a.store.get_run(a.source.run_id).state == "completed"
        assert a.store.frame_results(a.source.run_id, "iou")[0]["raw_value"] == 0.0
    # Simulate the previous shared-index layout while preserving all canonical events/blobs.
    legacy = a.store.layout.index_path
    with closing(sqlite3.connect(a.store.database)) as source, closing(sqlite3.connect(legacy)) as destination:
        source.backup(destination)
    a.store.database.unlink()
    migrated = FilesystemAnalysisStore(q.service.catalog_root, str(q.project.id))
    assert len(migrated.list_runs()) == 2
    assert migrated.frame_results(a.source.run_id, "iou")[0]["raw_value"] == 0.0
    assert migrated.frame_results(a.checked.run_id, "iou")[0]["raw_value"] == 1.0
    assert q.service.get_project(q.project.id) is not None
    assert migrated.get_run(a.source.run_id).state == "completed"


def test_orphan_legacy_analysis_cache_is_preserved_with_explicit_error(quality):
    q = quality
    from kraken_manager.infrastructure.filesystem.layout import FileProjectLayout
    layout = FileProjectLayout(q.service.catalog_root, str(q.project.id))
    with sqlite3.connect(layout.index_path) as connection:
        connection.execute("CREATE TABLE analysis_runs (run_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO analysis_runs VALUES ('legacy-without-canonical-event')")
    with pytest.raises(RuntimeError, match="old cache preserved"):
        FilesystemAnalysisStore(q.service.catalog_root, str(q.project.id))
    with sqlite3.connect(layout.index_path) as connection:
        assert connection.execute("SELECT run_id FROM analysis_runs").fetchone()[0] == "legacy-without-canonical-event"
    connection.close()
    assert not (layout.index_dir / "analysis.sqlite3").exists()
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore
    events = FilesystemEventStore(q.service.catalog_root, str(q.project.id))
    events.append("legacy-cache-check", expected_revision=0, events=({"event_type": "cache.checkpoint_changed", "payload": {}},))
    with pytest.raises(RuntimeError, match="old cache preserved"):
        q.service.get_project(q.project.id)
    with closing(sqlite3.connect(layout.index_path)) as connection:
        assert connection.execute("SELECT run_id FROM analysis_runs").fetchone()[0] == "legacy-without-canonical-event"


def test_legacy_import_without_canonical_result_is_not_silently_downgraded(analysis):
    a, q = analysis, analysis.q
    queued, _ = real_run(q, a.root / "unrecorded-import", a.store, a.target, a.reference, imported=False)
    legacy = a.store.layout.index_path
    with closing(sqlite3.connect(a.store.database)) as source, closing(sqlite3.connect(legacy)) as destination:
        source.backup(destination)
        with destination:
            destination.execute("UPDATE analysis_partitions SET state='imported', bundle_sha256=? WHERE run_id=?",
                                ("f" * 64, queued.run_id))
    a.store.database.unlink()
    with pytest.raises(RuntimeError, match="state/result lacks matching canonical"):
        FilesystemAnalysisStore(q.service.catalog_root, str(q.project.id))
    with closing(sqlite3.connect(legacy)) as connection:
        assert connection.execute("SELECT state FROM analysis_partitions WHERE run_id=?", (queued.run_id,)).fetchone()[0] == "imported"
    assert not a.store.database.exists()
