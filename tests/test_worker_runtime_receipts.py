"""Receipt contract/fault tests; numerical execution is covered by actual worker gates."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from karakal.runtime_identity import worker_runtime_identity
from karakal.worker import execute
from kraken_core.analysis_runtime import validate_worker_runtime
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
from tests.test_analysis_filesystem_store import _contracts, _result_bundle


def test_receipt_measures_loaded_libraries_and_changes_with_loaded_code(monkeypatch):
    from karakal.core import headless
    from kraken_core.plugins import load_plugin_metadata_from_directory
    from pathlib import Path

    before = worker_runtime_identity()
    assert before.numpy_version == np.__version__ and before.opencv_version == cv2.__version__
    assert before.engine_build.startswith("sha256:")
    assert load_plugin_metadata_from_directory(Path(__file__).resolve().parents[1] / "plugins/karakal").version == before.engine_version

    def altered(first, second):
        return {"xor": 0.1234}

    with monkeypatch.context() as scoped:
        scoped.setattr(headless._metric_values, "__code__", altered.__code__)
        assert worker_runtime_identity().engine_build != before.engine_build
    assert worker_runtime_identity() == before


def test_default_values_and_protocol_sets_affect_computation_fingerprint(monkeypatch):
    from kraken_core.analysis_protocol import AnalysisFrameResult
    from kraken_core import analysis_runtime, analysis_run_protocol

    before = worker_runtime_identity()
    with monkeypatch.context() as scoped:
        scoped.setattr(AnalysisFrameResult.__init__, "__defaults__", ((), "changed-default"))
        assert AnalysisFrameResult("frame", 1, 1, "ready", ()).message == "changed-default"
        assert worker_runtime_identity() != before
    with monkeypatch.context() as scoped:
        scoped.setattr(analysis_runtime.measured_runtime, "__kwdefaults__", {"require_all": False})
        assert analysis_runtime.measured_runtime([{"worker_runtime": before.to_payload()}, {}]) == before
        assert worker_runtime_identity() != before
    with monkeypatch.context() as scoped:
        scoped.setattr(analysis_run_protocol, "_EXPRESSION_OPERATIONS", frozenset())
        assert worker_runtime_identity() != before
    assert worker_runtime_identity() == before


def test_property_accessor_behavior_changes_computation_fingerprint(monkeypatch):
    from kraken_core.analysis_run_protocol import AnalysisExpression
    before = worker_runtime_identity()

    def changed_accessor(self):
        return frozenset()

    with monkeypatch.context() as scoped:
        scoped.setattr(AnalysisExpression.source_keys.fget, "__code__", changed_accessor.__code__)
        assert AnalysisExpression.source("A").source_keys == frozenset()
        assert worker_runtime_identity().engine_build != before.engine_build
    assert worker_runtime_identity() == before


def test_retry_uses_learned_worker_pin_after_first_partition_import(tmp_path):
    from tests.test_analysis_run_workflow import _coordinator
    coordinator, store, gateway, run = _coordinator(tmp_path, count=1001)
    first, second = store.all_partitions(run.run_id)
    first_result, first_bundle = gateway.results[first.job_id]
    store.import_partition(first_result, first_bundle)
    store.mark_partition_state(second.partition_id, "failed", error="Controlled retry fixture")
    assert second.expected_runtime is None
    coordinator.retry_failed(run.run_id)
    retried = store.all_partitions(run.run_id)[1]
    assert retried.expected_runtime == store.pinned_runtime(run.run_id) == first_result.worker_runtime
    assert gateway.manifests[retried.job_id].expected_runtime == first_result.worker_runtime


def test_computation_fingerprint_is_stable_across_independent_python_hash_seeds():
    code = "from karakal.runtime_identity import worker_runtime_identity;import json;print(json.dumps(worker_runtime_identity().to_payload()))"
    receipts = [json.loads(subprocess.check_output([sys.executable, "-c", code], text=True, timeout=20,
        env=dict(os.environ, PYTHONHASHSEED=str(seed)),
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))) for seed in (1, 2)]
    assert receipts[0] == receipts[1] == worker_runtime_identity().to_payload()


def test_native_worker_refuses_explicit_incompatible_runtime_before_reading_inputs(tmp_path):
    _run, partitions = _contracts(1)
    measured = worker_runtime_identity()
    job = replace(partitions[0], require_worker_runtime=True,
                  expected_runtime=replace(measured, numpy_version="incompatible-runtime-fixture"))
    job.write(tmp_path / "job.json")  # Input files deliberately do not exist.
    result = execute(tmp_path / "job.json", tmp_path / "result.json", tmp_path / "progress.json", tmp_path)
    assert result.outcome.value == "failed" and result.bundle is None
    assert result.worker_runtime == measured
    assert "pinned requested runtime" in result.message
    assert not (tmp_path / "outputs").exists()
    validate_worker_runtime(job, result)  # Refusal itself retains its measured environment.


def test_new_managed_import_requires_receipt_but_legacy_result_remains_readable(tmp_path):
    run, parts = _contracts(1)
    modern = replace(parts[0], require_worker_runtime=True)
    store = FilesystemAnalysisStore(tmp_path / "modern", run.project_id)
    store.create_run(run, (modern,))
    path, result = _result_bundle(tmp_path, modern)
    with pytest.raises(ValueError, match="receipt is missing"):
        store.import_partition(result, path)
    assert not store.frame_results(run.run_id, "iou")
    legacy = FilesystemAnalysisStore(tmp_path / "legacy", run.project_id)
    legacy.create_run(run, parts)
    legacy.import_partition(result, path)
    legacy.rebuild()
    assert legacy.get_run(run.run_id).worker_runtime is None
    assert len(legacy.frame_results(run.run_id, "iou")) == 1


def test_second_partition_with_different_worker_is_rejected_without_importing_its_frames(tmp_path):
    run, parts = _contracts(1001)
    parts = tuple(replace(part, require_worker_runtime=True) for part in parts)
    store = FilesystemAnalysisStore(tmp_path / "catalog", run.project_id)
    store.create_run(run, parts)
    runtime = worker_runtime_identity()
    first_path, first = _result_bundle(tmp_path, parts[0])
    store.import_partition(replace(first, worker_runtime=runtime), first_path)
    second_path, second = _result_bundle(tmp_path, parts[1])
    with pytest.raises(ValueError, match="incompatible worker runtimes"):
        store.import_partition(replace(second, worker_runtime=replace(runtime, python_version="another-fixture-python")), second_path)
    assert len(store.frame_results(run.run_id, "iou")) == 1000
    store.rebuild()
    assert store.get_run(run.run_id).worker_runtime == runtime
    assert store.get_run(run.run_id).state == "running"


def test_legacy_mixed_receipts_do_not_make_server_report_fully_measured_runtime():
    from kraken_server.persistent_services_parts.analysis_runs import AnalysisRunsMixin
    run, parts = _contracts(1001)
    runtime = worker_runtime_identity().to_payload()
    state = {"manifest": run.to_payload(), "layer_id": "layer", "state": "completed", "revision": 3,
             "actor_id": "actor", "resources": {}, "worker_runtime": runtime,
             "partitions": {part.partition_id: {"manifest": part.to_payload(), "state": "imported",
                 "result": {} if index == 0 else {"worker_runtime": runtime},
                 "completed_frames": len(part.frames), "failed_frames": 0} for index, part in enumerate(parts)}}
    report = AnalysisRunsMixin()._analysis_view(state)
    assert report["pinned_runtime"] == runtime
    assert report["worker_runtime"] is None and report["runtime_verification"] == "unverified"


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_runtime_requirement_cannot_be_downgraded_by_nonboolean_payload(value):
    _run, parts = _contracts(1)
    with pytest.raises(ValueError, match="boolean"):
        AnalysisPartitionJobManifest.from_payload(parts[0].to_payload() | {"require_worker_runtime": value})
