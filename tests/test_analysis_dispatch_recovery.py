"""Recover persisted dispatch intents without resurrecting cancelled runs."""
from kraken_manager.application.analysis_runs import AnalysisRunCoordinator
from tests.test_analysis_run_workflow import _coordinator
import pytest


def test_refresh_delivers_persisted_intents_after_creation_crash_without_new_ids(tmp_path):
    coordinator, store, gateway, run = _coordinator(tmp_path, count=1001)
    # A separate run is persisted exactly as start() does, before any Agent call.
    previous = store.all_partitions(run.run_id)
    recovered, partitions = coordinator.build_contracts(
        project_id=run.project_id, frames=tuple(frame for p in previous for frame in p.frames),
        source_bindings=run.source_bindings, recipe=run.recipe, runtime=run.runtime,
    )
    store.create_run(recovered, partitions)
    reopened = AnalysisRunCoordinator(store, gateway, coordinator.source_path_for_version)
    gateway.reject_submissions = True
    reopened.refresh(recovered.run_id)
    assert len(store.queued_partitions(recovered.run_id)) == 2
    gateway.reject_submissions = False
    report = reopened.refresh(recovered.run_id)
    assert report.state == "completed"
    assert {p.job_id for p in partitions} <= gateway.manifests.keys()
    deliveries = set(gateway.manifests)
    reopened.refresh(recovered.run_id)
    assert set(gateway.manifests) == deliveries


def test_cancel_before_agent_handoff_stays_cancelled_during_refresh(tmp_path):
    coordinator, store, gateway, run = _coordinator(tmp_path, count=1)
    previous = store.all_partitions(run.run_id)
    pending, partitions = coordinator.build_contracts(
        project_id=run.project_id, frames=previous[0].frames, source_bindings=run.source_bindings,
        recipe=run.recipe, runtime=run.runtime,
    )
    store.create_run(pending, partitions)
    coordinator.cancel(pending.run_id)
    assert store.get_run(pending.run_id).state == "cancelled"
    coordinator.refresh(pending.run_id)
    assert store.get_run(pending.run_id).state == "cancelled"
    assert partitions[0].job_id not in gateway.manifests


def test_cancel_survives_transport_outage_and_refuses_late_import(tmp_path, monkeypatch):
    coordinator, store, gateway, run = _coordinator(tmp_path, count=1)
    manifest, = store.all_partitions(run.run_id)
    result, bundle = gateway.results[manifest.job_id]

    def offline(*_args):
        raise OSError("Agent unavailable while cancelling")

    with monkeypatch.context() as scoped:
        scoped.setattr(gateway, "cancel", offline)
        coordinator.cancel(run.run_id)
    assert store.get_run(run.run_id).state == "cancelled"
    with pytest.raises(ValueError, match="late result"):
        store.import_partition(result, bundle)
    store.rebuild()
    restarted = AnalysisRunCoordinator(store, gateway, coordinator.source_path_for_version)
    restarted.refresh(run.run_id)
    assert gateway.get(manifest.job_id).state == "cancelled"
    assert store.get_run(run.run_id).state == "cancelled"
    assert not store.frame_results(run.run_id, "iou")
