from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_manager.application.lifecycle import ArchiveLayerHandler
from kraken_manager.domain.identity import Principal, ProjectRole
from kraken_manager.workspace import LayerSourceMode
from kraken_server.app import create_app
from kraken_server.services import (
    CommandContext,
    ConflictError,
    ForbiddenError,
    InMemoryServerServices,
    ValidationError,
)
from test_server_workspace_inputs_parts.server_layer import _server_layer


def deletion_setup(tmp_path, *, managed=False):
    server, local, session, project, layer, binding, *_ = _server_layer(tmp_path)
    layer = next(item for item in local.list_layers(project.id) if item.id == layer.id)
    server.profiles, server.identities = local.profiles, local.identities
    server.uow_factory = local._uow(str(project.id))
    server._archive_layer = ArchiveLayerHandler(local._uow(str(project.id)), local.profiles, local.clock)
    from kraken_manager.application.layer_deletion import DeleteLayerHandler
    from kraken_manager.infrastructure.layer_deletion import WorkspaceLayerDeletionStorage

    server._delete_layer = lambda command: DeleteLayerHandler(
        server.uow_factory,
        server.profiles,
        local.clock,
        WorkspaceLayerDeletionStorage(local.workspace_files),
        archive=server._archive_layer,
    )(command)
    if managed:
        workspace = local.workspace_files.registry.get_project(str(project.id))
        images = Path(workspace.source_project_dir) / "img" / layer.name
        images.mkdir(parents=True)
        (images / "source.jpg").write_bytes(b"source")
        binding = replace(binding, mode=LayerSourceMode.MANAGED_COPY, image_directory=str(images))
        local.workspace_files.registry.save_layer(str(project.id), binding)
    from kraken_manager.application.project_commands import ProjectCommandService

    server.commands = ProjectCommandService(
        server.uow_factory,
        local.profiles,
        local.clock,
        deletion_storage=WorkspaceLayerDeletionStorage(local.workspace_files),
    )
    context = CommandContext(str(session.principal.id), "../../untrusted-client-key", layer.revision)
    return server, local, session, project, layer, binding, context


@pytest.mark.parametrize("managed", [False, True])
def test_remote_layer_deletion_routes_to_server_and_replays(tmp_path, monkeypatch, managed):
    server, local, session, project, layer, binding, context = deletion_setup(tmp_path, managed=managed)
    remote = object.__new__(RemoteServerProjectService)
    calls = []

    def call(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return server.execute_project_command(
            str(project.id), kwargs["payload"], CommandContext(context.actor_id, kwargs["idempotency_key"], None)
        )

    remote._call = call
    dual = object.__new__(DualCatalogService)
    dual.local, dual.remote = local, remote
    dual._remote_ids = {str(project.id)}
    monkeypatch.setattr(local, "delete_layer", lambda **_kw: pytest.fail("Deleted a remote layer locally"))
    arguments = dict(
        principal=session.principal,
        project=project,
        layer=layer,
        confirmation_name=layer.name,
        idempotency_key=context.idempotency_key,
    )
    deleted = dual.delete_layer(**arguments)
    assert deleted.state.value == "archived"
    assert Path(binding.image_directory).exists() is (not managed)
    assert local.workspace_files.registry.get_layer(str(project.id), str(layer.id)) is None
    history = local.history(project.id)
    assert dual.delete_layer(**arguments) == deleted
    assert local.history(project.id) == history
    assert calls[0][0:2] == ("POST", f"/api/v1/projects/{project.id}/commands")
    assert calls[0][2]["payload"]["arguments"]["expected_revision"] == layer.revision


@pytest.mark.parametrize("failure", ["name", "revision", "job", "run", "permission", "archive"])
def test_deletion_failure_preserves_files_and_binding(tmp_path, monkeypatch, failure):
    server, local, session, project, layer, binding, context = deletion_setup(tmp_path, managed=True)
    name, error = layer.name, ConflictError
    if failure == "name":
        name, error = "wrong", ValidationError
    elif failure == "revision":
        context = replace(context, expected_revision=layer.revision + 1)
    elif failure == "job":
        factory = server.uow_factory

        def active_job_uow():
            unit = factory()
            unit.projections.list_plugin_jobs = lambda _id: (
                SimpleNamespace(layer_id=layer.id, state=SimpleNamespace(value="running")),
            )
            return unit

        server.uow_factory = active_job_uow
    elif failure == "run":
        monkeypatch.setattr(
            local.workspace_files.registry,
            "list_runs",
            lambda *_args: (SimpleNamespace(state=SimpleNamespace(value="running")),),
        )
    elif failure == "permission":
        viewer = Principal.local(subject="viewer", display_name="Viewer")
        local.identities.save(viewer)
        local.assign_project_role(
            principal=session.principal,
            project=project,
            target_principal_id=viewer.id,
            role=ProjectRole.VIEWER,
            expected_revision=0,
            idempotency_key="viewer",
        )
        server._actor = lambda _id: viewer
        error = ForbiddenError
    else:

        def fail_archive(*_args):
            assert not Path(binding.image_directory).exists()
            raise ConflictError("Concurrent change")

        monkeypatch.setattr(server, "_archive_layer", fail_archive)
    with pytest.raises(error):
        server.delete_layer(str(project.id), str(layer.id), name, context)
    assert Path(binding.image_directory, "source.jpg").read_bytes() == b"source"
    assert local.workspace_files.registry.get_layer(str(project.id), str(layer.id)) == binding
    assert local.get_project(project.id) is not None
    assert local.list_layers(project.id)[0].state.value == "active"


def test_delete_replay_completes_failed_registry_cleanup(tmp_path, monkeypatch):
    server, local, _session, project, layer, binding, context = deletion_setup(tmp_path, managed=True)
    registry = local.workspace_files.registry
    remove_layer = registry.remove_layer

    def fail_cleanup(*_args):
        raise OSError("Registry unavailable")

    monkeypatch.setattr(registry, "remove_layer", fail_cleanup)
    with pytest.raises(OSError):
        server.delete_layer(str(project.id), str(layer.id), layer.name, context)
    assert not Path(binding.image_directory).exists()
    assert registry.get_layer(str(project.id), str(layer.id)) == binding
    history = local.history(project.id)
    monkeypatch.setattr(registry, "remove_layer", remove_layer)
    assert server.delete_layer(str(project.id), str(layer.id), layer.name, context)["state"] == "archived"
    assert registry.get_layer(str(project.id), str(layer.id)) is None
    assert local.history(project.id) == history


def test_concurrent_deletion_cannot_restore_an_archived_layers_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    server, local, _session, project, layer, binding, context = deletion_setup(tmp_path, managed=True)
    start = Barrier(2)

    def delete(key):
        start.wait()
        try:
            server.delete_layer(str(project.id), str(layer.id), layer.name, replace(context, idempotency_key=key))
            return "deleted"
        except ConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(delete, ("delete-a", "delete-b")))
    assert sorted(results) == ["conflict", "deleted"]
    assert not Path(binding.image_directory).exists()
    assert local.list_layers(project.id) == ()


def test_delete_api_requires_auth_confirmation_and_revision():
    calls = []

    class Services(InMemoryServerServices):
        def delete_layer(self, *args):
            calls.append(args)
            return {"state": "archived"}

    client = TestClient(create_app(services=Services(), development=True))
    url = "/api/v1/projects/project/layers/layer/delete"
    headers = {"Authorization": "Bearer developer", "Idempotency-Key": "delete", "If-Match": "3"}
    assert client.post(url, json={"confirmation_name": "Metal"}).status_code == 401
    assert (
        client.post(
            url,
            json={"confirmation_name": "Metal"},
            headers={key: value for key, value in headers.items() if key != "If-Match"},
        ).status_code
        == 422
    )
    for value in (None, [], 42, ""):
        assert client.post(url, json={"confirmation_name": value}, headers=headers).status_code == 422
    assert not calls
    assert client.post(url, json={"confirmation_name": "Metal"}, headers=headers).status_code == 200
    assert calls[0][:3] == ("project", "layer", "Metal")
    assert calls[0][3].expected_revision == 3
