"""test_server_workspace_inputs: test task pipeline removal."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_server.services import CommandContext

from test_server_workspace_inputs_parts.desktop_job import _desktop_job

def test_task_and_pipeline_removal_preserve_audit_and_can_be_repeated(tmp_path):
    from kraken_manager.infrastructure.filesystem._codec import decode_model
    from kraken_manager.domain.workflows import PluginJob

    server, local, session, project, layer, _target, gateway, payload = _desktop_job(
        tmp_path, "frames.dataset.prepare.v1"
    )
    job = decode_model(PluginJob, payload)
    creation = next(event for event in local.history(project.id) if event.event_type == "PluginJobCreated")
    with pytest.raises(ValueError, match="Cancel"):
        local.remove_layer_pipeline_action(
            principal=session.principal,
            project_id=project.id,
            layer_id=layer.id,
            action_event_id=creation.event_id,
            remove_task=True,
        )
    controller = object.__new__(DesktopController)
    controller.service = local
    controller.session = session
    controller._project_id = project.id
    controller.my_work_refresh = None
    controller._gateway_for_job = lambda _job: gateway
    snapshot = controller._pipeline_snapshot(project.id, layer.id)
    nodes = [node for lane in snapshot.lanes for node in lane.nodes]
    assert len([node for node in nodes if node.kind == "job"]) == 1
    assert not any(node.kind == "dataset" for node in nodes)
    assert next(node for node in nodes if node.kind == "job").details["deletable"]
    local.remove_layer_pipeline_action(
        principal=session.principal, project_id=project.id, layer_id=layer.id, action_event_id=creation.event_id
    )
    assert local.plugin_jobs()
    assert not any(
        node.kind == "job" for lane in controller._pipeline_snapshot(project.id, layer.id).lanes for node in lane.nodes
    )
    controller.delete_agent_job(job)
    assert not local.plugin_jobs()
    assert any(event.event_id == creation.event_id for event in local.history(project.id))
    removed = local.remove_layer_pipeline_action(
        principal=session.principal,
        project_id=project.id,
        layer_id=layer.id,
        action_event_id=creation.event_id,
        remove_task=True,
    )
    assert removed.payload["remove_task"] is True
    # Exercise the server facade against the same real domain projections.
    events = local.history(project.id)
    server.events = SimpleNamespace(
        list_project_events=lambda _id, **kwargs: tuple(
            (index, event) for index, event in enumerate(events, 1) if index > kwargs["after_position"]
        )
    )
    projections = server.projections
    server.projections = SimpleNamespace(
        list_projects=lambda **_kwargs: (project,),
        list_plugin_jobs=projections.list_plugin_jobs,
        get_layer=projections.get_layer,
        get_plugin_job=projections.get_plugin_job,
    )
    assert server.list_plugin_jobs(str(project.id)) == []
    server._append_auxiliary_event = lambda _id, **kwargs: kwargs["payload"]
    result = server.append_pipeline_event(
        str(project.id),
        {
            "event_type": "LayerPipelineActionRemoved",
            "layer_id": str(layer.id),
            "action_event_id": creation.event_id,
            "remove_task": True,
            "plugin_job_id": "spoofed",
        },
        CommandContext(str(session.principal.id), "remove", None),
    )
    assert result["plugin_job_id"] == str(job.id)
