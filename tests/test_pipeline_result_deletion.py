"""Removing a pipeline step removes only that job's materialized files."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kraken_server.services import CommandContext
from test_server_workspace_inputs_parts.desktop_job import _desktop_job


@pytest.mark.parametrize("backend", ["local", "server"])
def test_step_removal_cleans_recorded_result_and_can_retry(tmp_path, backend):
    server, local, session, project, layer, _target, _gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    source_root, derived_root = tmp_path / "source-root", tmp_path / "derived-root"
    source_root.mkdir()
    derived_root.mkdir()
    binding = local.bind_project_workspace(
        principal=session.principal,
        project=project,
        source_root=source_root,
        derived_root=derived_root,
        idempotency_key="workspace",
    )
    relative = Path("vector") / layer.name / "Contour result"
    output = Path(binding.derived_project_dir) / relative
    output.mkdir(parents=True)
    (output / "0000.cif").write_bytes(b"CIF")
    local.workspace_registry.save_plugin_result_directory(str(project.id), job["id"], str(relative))
    creation = next(event for event in local.history(project.id) if event.event_type == "PluginJobCreated")
    if backend == "local":

        def remove():
            return local.remove_layer_pipeline_action(
                principal=session.principal,
                project_id=project.id,
                layer_id=layer.id,
                action_event_id=creation.event_id,
            )
    else:
        server._pipeline_events = lambda _pid: (creation,)
        server._append_auxiliary_event = lambda _pid, **kwargs: kwargs["payload"]

        def remove():
            return server.append_pipeline_event(
                str(project.id),
                {
                    "event_type": "LayerPipelineActionRemoved",
                    "layer_id": str(layer.id),
                    "action_event_id": creation.event_id,
                    "plugin_job_id": "spoofed",
                },
                CommandContext(str(session.principal.id), "remove", None),
            )

    remove()
    assert not output.exists()
    remove()
    assert local.workspace_registry.plugin_result_removed(str(project.id), job["id"])
    assert local.plugin_jobs()  # Removing just the step still retains task history.


def test_failed_server_command_does_not_remove_files(tmp_path):
    server, local, _session, project, layer, _target, _gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    creation = next(event for event in local.history(project.id) if event.event_type == "PluginJobCreated")
    server._pipeline_events = lambda _pid: (creation,)
    calls = []
    server.workspace_files = SimpleNamespace(registry=SimpleNamespace(mutation_lock=lambda _pid: calls.append(True)))

    def reject(*_args, **_kwargs):
        raise PermissionError("Denied")

    server._append_auxiliary_event = reject
    with pytest.raises(PermissionError, match="Denied"):
        server.append_pipeline_event(
            str(project.id),
            {
                "event_type": "LayerPipelineActionRemoved",
                "layer_id": str(layer.id),
                "action_event_id": creation.event_id,
            },
            CommandContext("actor", "remove", None),
        )
    assert not calls
