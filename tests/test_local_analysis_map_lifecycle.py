"""Local canonical map intent, cancellation and immutable input guards."""
from dataclasses import replace
import hashlib
from uuid import uuid4

import pytest

from kraken_core.analysis_protocol import AnalysisParameter
from kraken_core.analysis_run_protocol import AnalysisPartitionJobManifest, AnalysisRenderedMap
from tests.test_analysis_run_workflow import _coordinator
from tests.test_electrical_graph import png


def pending_map(tmp_path):
    coordinator, store, gateway, run = _coordinator(tmp_path, count=1)
    coordinator.refresh(run.run_id)
    gateway.reject_submissions = True
    frame_id = run.frame_ids[0]
    task = coordinator.request_map(run.run_id, frame_id, idempotency_key="pending-map")
    assert task["state"] == "queued"
    return coordinator, store, gateway, run, frame_id, task


def test_cancelled_map_survives_replay_and_rejects_late_result(tmp_path):
    coordinator, store, gateway, run, frame_id, task = pending_map(tmp_path)
    manifest = AnalysisPartitionJobManifest.from_payload(task["manifest"])
    original_result, _bundle = next(iter(gateway.results.values()))
    coordinator.cancel_maps(run.run_id)
    store.rebuild()
    assert store.map_tasks(run.run_id)[frame_id]["state"] == "cancelled"
    data = png([[255, 0]])
    result = replace(original_result, job_id=manifest.job_id, partition_id=manifest.partition_id,
                     rendered_map=AnalysisRenderedMap(frame_id, "outputs/map.png", hashlib.sha256(data).hexdigest(), len(data)))
    with pytest.raises(ValueError, match="late result"):
        store.import_map(manifest, result, data)
    assert store.get_run(run.run_id).state == "completed"
    assert len(store.frame_results(run.run_id, "iou")) == 1


def test_map_request_refuses_changed_input_or_parameters(tmp_path):
    _coordinator_object, store, _gateway, run, _frame_id, task = pending_map(tmp_path)
    manifest = AnalysisPartitionJobManifest.from_payload(task["manifest"])
    for forged in (
        replace(manifest, job_id=str(uuid4()), parameters=manifest.parameters + (AnalysisParameter("threshold", 0.9),)),
        replace(manifest, job_id=str(uuid4()), frames=(replace(manifest.frames[0], x=2),)),
    ):
        with pytest.raises(ValueError, match="pinned"):
            store.request_map(forged)
    assert len(store.map_tasks(run.run_id)) == 1
