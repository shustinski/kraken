from uuid import uuid4

import pytest

from kraken_agent.jobs import AgentJobState
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.domain.pipeline import LayerPipelineDefinition
from kraken_manager.domain.pipeline_layer import PipelineLayerOperation
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_manager.infrastructure.plugin.agent_gateway import AgentUnavailable
from tests.test_stitch_local_workflow import computing as computing
from tests.test_stitch_plan_workflow import geometry as geometry


def define(computing):
    local, session, project, *_ = computing
    model = MultilayerModel.from_dict(local.get_multilayer_model(project.id))
    stitch = model.stitches[-1]
    operation = PipelineLayerOperation(str(uuid4()), "Layer mosaic", "layer.stitch.v1", "1.0.0", stitch.id, stitch.revision,
                                       parameters={"block_size": 8})
    definition = LayerPipelineDefinition(str(project.id), stitch.layer_id, (operation,))
    assert LayerPipelineDefinition.from_dict(definition.to_dict()).fingerprint == definition.fingerprint
    assert "target_representation_id" not in operation.to_dict() and "coordinates" not in operation.to_dict()
    revision = FilesystemEventStore(local.catalog_root, project.id).current_revision(f"layer-pipeline:{stitch.layer_id}")
    state = local.define_layer_pipeline(project_id=project.id, layer_id=stitch.layer_id, definition=definition.to_dict(),
        principal=session.principal, expected_revision=revision, idempotency_key="definition")
    state = local.command_layer_pipeline(project_id=project.id, layer_id=stitch.layer_id, operation="start",
        principal=session.principal, expected_revision=state["revision"], idempotency_key="start")
    return operation, state, dict(project_id=project.id, layer_id=stitch.layer_id, principal=session.principal)


def test_native_layer_attempt_is_imported_in_the_existing_pipeline_and_reopens(computing):
    local, _, project, _, _, _, gateway, runner, control = computing
    operation, _, args = define(computing)
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    attempt = state["nodes"][operation.id]["attempts"][-1]
    assert attempt["request"]["kind"] == "layer"
    render = next(r for r in MultilayerModel.from_dict(local.get_multilayer_model(project.id)).stitch_renders if r.job_id == attempt["job_id"])
    assert render.state == "requested" and control.store.get(render.job_id).state is AgentJobState.QUEUED
    assert runner.run_once()
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    assert state["status"] == "completed"
    output_id = state["nodes"][operation.id]["attempts"][-1]["output_version_ids"][0]
    output = local.get_artifact_version(project.id, output_id)
    assert local._projection(project.id).get_artifact_series(output.series_id).frame_id is None
    reopened = EmbeddedProjectService(local.data_dir)
    assert reopened.get_layer_pipeline(project.id, args["layer_id"]) == state
    assert control.store.get(render.job_id).state is AgentJobState.SUCCEEDED


def test_pause_prevents_new_native_handoff_then_resume_uses_the_same_attempt(computing, monkeypatch):
    local, _, project, _, _, _, gateway, runner, control = computing
    operation, _, args = define(computing)
    submit = gateway.submit
    monkeypatch.setattr(gateway, "submit", lambda *_: (_ for _ in ()).throw(AgentUnavailable("controlled outage before enqueue")))
    with pytest.raises(AgentUnavailable):
        local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    state = local.get_layer_pipeline(project.id, args["layer_id"])
    job_id = state["nodes"][operation.id]["attempts"][-1]["job_id"]
    local.command_layer_pipeline(**args, operation="pause", expected_revision=state["revision"], idempotency_key="pause")
    monkeypatch.setattr(gateway, "submit", submit)
    local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    with pytest.raises(KeyError):
        control.store.get(job_id)
    state = local.get_layer_pipeline(project.id, args["layer_id"])
    local.command_layer_pipeline(**args, operation="resume", expected_revision=state["revision"], idempotency_key="resume")
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    assert len(state["nodes"][operation.id]["attempts"]) == 1
    assert control.store.get(job_id).state is AgentJobState.QUEUED
    assert runner.run_once()
    assert local.tick_layer_pipeline(**args, stitch_gateway=gateway)["status"] == "completed"


def test_pipeline_cancel_durably_fences_late_native_result(computing):
    from kraken_core.plugin_protocol import PluginJobManifestV2
    local, session, project, _, _, _, gateway, runner, control = computing
    operation, _, args = define(computing)
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    job_id = state["nodes"][operation.id]["attempts"][-1]["job_id"]
    assert runner.run_once()
    render = next(r for r in MultilayerModel.from_dict(local.get_multilayer_model(project.id)).stitch_renders if r.job_id == job_id)
    publication, path = gateway.result(PluginJobManifestV2.from_json(render.manifest_json))
    local.command_layer_pipeline(**args, operation="cancel", expected_revision=state["revision"], idempotency_key="cancel")
    with pytest.raises(ConflictError, match="late result"):
        local.import_stitch_publication(project_id=project.id, principal=session.principal, publication=publication, source_path=path)
    state = local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    assert state["status"] == "cancelled" and control.store.get(job_id).state is AgentJobState.CANCELLED


def test_reservation_and_physical_request_roll_back_together(computing, monkeypatch):
    from kraken_manager.infrastructure.filesystem.unit_of_work import _StagedEventStore
    local, _, project, _, _, _, gateway, _, _ = computing
    _, initial, args = define(computing)
    before = local.get_multilayer_model(project.id)
    append = _StagedEventStore.append

    def fail(self, stream_id, *, expected_revision, events):
        result = append(self, stream_id, expected_revision=expected_revision, events=events)
        if events[-1].payload.get("operation") == "pipeline_stitch_requested":
            raise OSError("controlled reservation transaction failure")
        return result

    monkeypatch.setattr(_StagedEventStore, "append", fail)
    with pytest.raises(OSError, match="reservation transaction"):
        local.tick_layer_pipeline(**args, stitch_gateway=gateway)
    assert local.get_layer_pipeline(project.id, args["layer_id"]) == initial
    assert local.get_multilayer_model(project.id) == before
