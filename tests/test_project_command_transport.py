"""One command and result contract across direct calls and authenticated HTTP."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_manager.application.dto import (
    CommandContext,
    CreateLayerCommand,
    RenameLayerCommand,
    ReorderLayersCommand,
)
from kraken_manager.application.project_commands import ProjectCommandService
from kraken_manager.domain.common import LayerId
from kraken_manager.domain.identity import Principal, ProjectRole
from kraken_manager.domain.project import Layer
from kraken_manager.infrastructure.project_command_codec import encode_project_command
from kraken_manager.infrastructure.workspace_rename import FilesystemWorkspaceRenameStorage
from kraken_server.app import create_app
from kraken_server.services import InMemoryServerServices
from test_catalog_operation_parity import import_setup


def command_transport(tmp_path):
    server, local, session, project, layer, _representation, _context, _root = import_setup(tmp_path)
    server.commands = ProjectCommandService(
        server.uow_factory,
        local.profiles,
        local.clock,
        rename_storage=FilesystemWorkspaceRenameStorage(local.workspace_files),
    )

    class Services(InMemoryServerServices):
        def execute_project_command(self, *args):
            return server.execute_project_command(*args)

    client = TestClient(create_app(services=Services(), development=True))
    remote = object.__new__(RemoteServerProjectService)

    def call(method, path, *, payload, idempotency_key):
        response = client.request(
            method,
            path,
            json=payload,
            headers={
                "Authorization": "Bearer developer",
                "Idempotency-Key": idempotency_key,
            },
        )
        if response.status_code >= 400:
            from kraken_hub.application.remote_error import RemoteServerError

            raise RemoteServerError(response.text, status=response.status_code)
        return response.json()

    remote._call = call
    return server, local, session, project, layer, client, remote


@pytest.mark.parametrize("transport", ["direct", "http"])
def test_same_creation_command_and_domain_result_use_both_transports(tmp_path, transport):
    _server, local, session, project, _layer, _client, remote = command_transport(tmp_path)
    current = local.get_project(project.id)
    command = CreateLayerCommand(
        context=CommandContext(actor=session.principal, idempotency_key="command-layer"),
        project_id=project.id,
        name="New layer",
        type=local.list_layers(project.id)[0].type,
        order=2,
        expected_project_revision=current.revision,
    )
    gateway = local if transport == "direct" else remote
    layer = gateway.execute_project_command(command)
    assert isinstance(layer, Layer)
    assert layer.id == command.layer_id
    assert layer.name == "New layer"
    history = local.history(project.id)
    assert gateway.execute_project_command(command) == layer
    assert local.history(project.id) == history


def test_http_uses_authenticated_identity_instead_of_the_callers_context(tmp_path):
    _server, local, session, project, layer, _client, remote = command_transport(tmp_path)
    impostor = Principal.local(subject="impostor", display_name="Impostor")
    current = next(item for item in local.list_layers(project.id) if item.id == layer.id)
    result = remote.execute_project_command(
        RenameLayerCommand(
            context=CommandContext(actor=impostor, idempotency_key="rename"),
            project_id=project.id,
            layer_id=layer.id,
            name="Renamed by owner",
            expected_revision=current.revision,
        )
    )
    assert result.name == "Renamed by owner"
    event = next(item for item in local.history(project.id) if item.event_type == "LayerRenamed")
    assert event.actor.principal_id == session.principal.id


def test_http_reorder_decodes_identifier_revision_pairs(tmp_path):
    _server, local, session, project, _layer, _client, remote = command_transport(tmp_path)
    layers = local.list_layers(project.id)
    command = ReorderLayersCommand(
        context=CommandContext(actor=session.principal, idempotency_key="reorder"),
        project_id=project.id,
        layer_ids=tuple(LayerId(str(item.id)) for item in reversed(layers)),
        expected_revisions=tuple((item.id, item.revision) for item in layers),
    )
    result = remote.execute_project_command(command)
    assert isinstance(result, tuple)
    assert all(isinstance(item, Layer) for item in result)
    assert tuple(item.id for item in result) == command.layer_ids


def test_wire_rejects_unauthenticated_protected_cross_project_and_internal_commands(tmp_path):
    _server, local, session, project, layer, client, _remote = command_transport(tmp_path)
    current = next(item for item in local.list_layers(project.id) if item.id == layer.id)
    command = RenameLayerCommand(
        context=CommandContext(actor=session.principal, idempotency_key="rename"),
        project_id=project.id,
        layer_id=layer.id,
        name="Unchanged",
        expected_revision=current.revision,
    )
    url = f"/api/v1/projects/{project.id}/commands"
    payload = encode_project_command(command)
    assert client.post(url, json=payload).status_code == 401
    headers = {"Authorization": "Bearer developer", "Idempotency-Key": "rename"}
    assert (
        client.post(
            url, json={**payload, "arguments": {**payload["arguments"], "context": {}}}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            url, json={**payload, "arguments": {**payload["arguments"], "project_id": str(uuid4())}}, headers=headers
        ).status_code
        == 422
    )
    for name in ("SynchronizePluginJobCommand", "AddArtifactVersionCommand", "CreateProjectCommand", "__import__"):
        assert client.post(url, json={**payload, "type": name}, headers=headers).status_code == 422
    invalid = {**payload, "arguments": {**payload["arguments"], "expected_revision": True}}
    assert client.post(url, json=invalid, headers=headers).status_code == 422
    assert local.list_layers(project.id)[0].name == layer.name


def test_wire_command_preserves_domain_authorization(tmp_path):
    server, local, session, project, layer, client, _remote = command_transport(tmp_path)
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
    server._actor = lambda _identifier: viewer
    current = next(item for item in local.list_layers(project.id) if item.id == layer.id)
    command = RenameLayerCommand(
        context=CommandContext(actor=session.principal, idempotency_key="denied"),
        project_id=project.id,
        layer_id=layer.id,
        name="Denied",
        expected_revision=current.revision,
    )
    response = client.post(
        f"/api/v1/projects/{project.id}/commands",
        json=encode_project_command(command),
        headers={"Authorization": "Bearer developer", "Idempotency-Key": "denied"},
    )
    assert response.status_code == 403
