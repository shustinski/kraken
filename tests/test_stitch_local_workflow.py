from dataclasses import replace
from pathlib import Path
import sys
import threading

import pytest

from kraken_agent.jobs import AgentJobState
from kraken_agent.runner import PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
from kraken_agent.service import AgentControlServer
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_core.plugin_protocol import PluginJobManifestV2
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.infrastructure.mosaic import MosaicReader, STITCH_OPERATION
from kraken_manager.infrastructure.plugin.agent_gateway import AgentUnavailable
from kraken_manager.infrastructure.plugin.stitch_gateway import AgentStitchGateway
from tests.test_stitch_plan_workflow import geometry as geometry


@pytest.fixture
def computing(geometry, tmp_path):
    local, session, project, _, versions, args, command = geometry
    payload = command("plan_layer_stitch", args)
    stitch = MultilayerModel.from_dict(payload).stitches[-1]
    coordinates = {str(project.frame_id_at(x, y)): [x, y] for x in range(1, 3) for y in range(1, 3)}
    wanted = {p.frame_id for p in stitch.tiles}
    requested = command("request_layer_stitch", {"stitch_id": stitch.id, "stitch_revision": stitch.revision,
        "block_size": 8, "frame_coordinates": {k: v for k, v in coordinates.items() if k in wanted}})
    request = MultilayerModel.from_dict(requested).stitch_renders[-1]
    control = AgentControlServer.create(tmp_path / "agent.sqlite3", token="stitch-test")
    httpd = control.build_http_server()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    gateway = AgentStitchGateway(base_url=f"http://127.0.0.1:{control.port}", token="stitch-test", staging_root=tmp_path / "staging")
    registry = PluginRegistry({STITCH_OPERATION: PluginProcessSpec(STITCH_OPERATION,
        (sys.executable, "-m", "kraken_manager.infrastructure.stitch_worker"))})
    runner = SubprocessPluginRunner(control.store, gateway.staging_root, registry,
        environment={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    try:
        yield local, session, project, versions, command, request, gateway, runner, control
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_full_local_http_agent_import_replay_reopen_and_lost_ack(computing, monkeypatch):
    local, session, project, versions, _, request, gateway, runner, control = computing
    arguments = dict(project_id=project.id, job_id=request.job_id, principal=session.principal, gateway=gateway)
    submit = gateway.submit

    def lost_submit(*args, **kwargs):
        submit(*args, **kwargs)
        raise AgentUnavailable("lost reply after durable enqueue")
    monkeypatch.setattr(gateway, "submit", lost_submit)
    with pytest.raises(AgentUnavailable):
        local.dispatch_stitch(**arguments)
    assert control.store.get(request.job_id).state is AgentJobState.QUEUED
    monkeypatch.setattr(gateway, "submit", submit)
    local.dispatch_stitch(**arguments)
    assert runner.run_once()
    assert control.store.get(request.job_id).state is AgentJobState.IMPORTING
    complete = gateway.complete_import
    monkeypatch.setattr(gateway, "complete_import", lambda *_: (_ for _ in ()).throw(AgentUnavailable("lost import ack")))
    with pytest.raises(AgentUnavailable):
        local.refresh_stitch(**arguments)
    model = MultilayerModel.from_dict(local.get_multilayer_model(project.id))
    imported = model.stitch_renders[-1]
    assert imported.state == "imported"
    version = local.get_artifact_version(project.id, imported.output.version_id)
    assert set(version.input_version_ids) == {model.stitches[-1].plan.version_id, *(v.id for v in versions)}
    series = local._projection(project.id).get_artifact_series(version.series_id)
    assert series.scope is ArtifactScope.LAYER_ATTACHMENT
    assert series.frame_id is None and series.representation_id is None
    with MosaicReader(local.artifact_source_path(project.id, version.id)) as reader:
        reader.validate(expected_plan=model.stitches[-1], expected_job_id=request.job_id)
        assert reader.read_block(0, 0).coverage.max() == 2
    history = local.history(project.id)
    monkeypatch.setattr(gateway, "complete_import", complete)
    local.refresh_stitch(**arguments)
    assert control.store.get(request.job_id).state is AgentJobState.SUCCEEDED
    assert local.history(project.id) == history
    reopened = EmbeddedProjectService(local.data_dir)
    assert reopened.get_multilayer_model(project.id) == local.get_multilayer_model(project.id)
    publication, path = gateway.result(PluginJobManifestV2.from_json(request.manifest_json))
    path.unlink()
    reopened.import_stitch_publication(project_id=project.id, principal=session.principal, publication=publication, source_path=None)
    assert local.history(project.id) == history
    # Remove only this fixture's disposable cache; canonical events rebuild it.
    local._projection(project.id).path.unlink()
    rebuilt = EmbeddedProjectService(local.data_dir)
    assert rebuilt.get_artifact_version(project.id, version.id) == version
    assert rebuilt.get_multilayer_model(project.id)["model"]["stitch_renders"][-1]["state"] == "imported"


def test_durable_cancel_refuses_late_result_and_repairs_agent_cancel(computing):
    local, session, project, _, command, request, gateway, runner, control = computing
    arguments = dict(project_id=project.id, job_id=request.job_id, principal=session.principal, gateway=gateway)
    local.dispatch_stitch(**arguments)
    assert runner.run_once()
    command("cancel_layer_stitch", {"job_id": request.job_id})
    publication, path = gateway.result(PluginJobManifestV2.from_json(request.manifest_json))
    with pytest.raises(ConflictError, match="late result"):
        local.import_stitch_publication(project_id=project.id, principal=session.principal, publication=publication, source_path=path)
    local.refresh_stitch(**arguments)
    assert control.store.get(request.job_id).state is AgentJobState.CANCELLED
    assert local.get_multilayer_model(project.id)["model"]["stitch_renders"][-1]["output"] is None


def test_import_refuses_stale_source_and_different_publication(computing, tmp_path):
    local, session, project, versions, _, request, gateway, runner, _ = computing
    arguments = dict(project_id=project.id, job_id=request.job_id, principal=session.principal, gateway=gateway)
    local.dispatch_stitch(**arguments)
    assert runner.run_once()
    changed = tmp_path / "changed.pgm"
    changed.write_bytes(b"P5\n1 1\n255\n\x00")
    local.add_managed_artifact_version(principal=session.principal, project_id=project.id, series_id=versions[0].series_id,
        source=changed, parent_version_id=versions[0].id, idempotency_key="replace-after-compute")
    with pytest.raises(ConflictError, match="stale"):
        local.refresh_stitch(**arguments)
    publication, path = gateway.result(PluginJobManifestV2.from_json(request.manifest_json))
    with pytest.raises(NotFoundError, match="not found"):
        local.import_stitch_publication(project_id=project.id, principal=session.principal,
            publication=replace(publication, job_id="unknown"), source_path=path)


def test_failed_local_transaction_rolls_back_artifact_and_receipt(computing, monkeypatch):
    from uuid import UUID, uuid5
    from kraken_manager.infrastructure.filesystem.unit_of_work import _StagedEventStore

    local, session, project, _, _, request, gateway, runner, _ = computing
    arguments = dict(project_id=project.id, job_id=request.job_id, principal=session.principal, gateway=gateway)
    local.dispatch_stitch(**arguments)
    assert runner.run_once()
    history = local.history(project.id)
    original = _StagedEventStore.append

    def reject(self, stream_id, *, expected_revision, events):
        result = original(self, stream_id, expected_revision=expected_revision, events=events)
        if events[-1].payload.get("operation") == "import_layer_stitch":
            raise OSError("injected local mosaic transaction failure")
        return result

    monkeypatch.setattr(_StagedEventStore, "append", reject)
    with pytest.raises(OSError, match="transaction failure"):
        local.refresh_stitch(**arguments)
    assert local.history(project.id) == history
    assert local._projection(project.id).get_artifact_version(str(uuid5(UUID(request.job_id), "mosaic-version"))) is None
    assert local._projection(project.id).get_artifact_series(str(uuid5(UUID(request.job_id), "mosaic-series"))) is None
    assert local.get_multilayer_model(project.id)["model"]["stitch_renders"][-1]["state"] == "requested"
    monkeypatch.setattr(_StagedEventStore, "append", original)
    assert local.refresh_stitch(**arguments)["model"]["stitch_renders"][-1]["state"] == "imported"
