from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_manager.domain.identity import Principal, ProjectRole
from kraken_manager.presentation.qt.layer_creation import LayerCreationDialog
from kraken_server.app import create_app
from kraken_server.services import CommandContext, ForbiddenError, InMemoryServerServices

from test_server_workspace_inputs import _server_layer


def test_server_scan_requires_structure_permission_and_uses_project_limit(tmp_path):
    server, local, session, project, _layer, binding, _representation, _context = _server_layer(tmp_path)
    server.profiles, server.identities = local.profiles, local.identities
    result = server.scan_layer_source(
        str(project.id), binding.image_directory, CommandContext(str(session.principal.id), "scan", None)
    )
    assert result["total_files"] == 2
    assert result["frame_positions"] == {"0000.jpg": 1, "0002.jpg": 3}
    viewer = Principal.local(subject="viewer", display_name="Viewer")
    local.identities.save(viewer)
    server._actor = lambda _id: viewer
    local.assign_project_role(
        principal=session.principal,
        project=project,
        target_principal_id=viewer.id,
        role=ProjectRole.VIEWER,
        expected_revision=0,
        idempotency_key="viewer",
    )
    with pytest.raises(ForbiddenError):
        server.scan_layer_source(
            str(project.id), binding.image_directory, CommandContext(str(viewer.id), "denied", None)
        )


def test_scan_api_requires_authentication_and_valid_directory():
    class Services(InMemoryServerServices):
        def scan_layer_source(self, project_id, directory, context):
            return {"project_id": project_id, "directory": directory, "actor": context.actor_id}

    client = TestClient(create_app(services=Services(), development=True))
    url = "/api/v1/projects/project/layers/scan"
    assert client.post(url, json={"directory": "source"}).status_code == 401
    headers = {"Authorization": "Bearer developer"}
    for directory in (None, "", [], 42):
        assert client.post(url, json={"directory": directory}, headers=headers).status_code == 422
    result = client.post(url, json={"directory": "server-only-source"}, headers=headers)
    assert result.status_code == 200
    assert result.json()["directory"] == "server-only-source"


def test_remote_scan_never_falls_back_to_local_scanner(tmp_path, monkeypatch):
    server, local, session, project, _layer, binding, _representation, _context = _server_layer(tmp_path)
    server.profiles, server.identities = local.profiles, local.identities
    remote = object.__new__(RemoteServerProjectService)
    calls = []

    def call(method, path, **kwargs):
        calls.append((method, path))
        return server.scan_layer_source(
            str(project.id), kwargs["payload"]["directory"], CommandContext(str(session.principal.id), "scan", None)
        )

    remote._call = call
    dual = object.__new__(DualCatalogService)
    dual.local, dual.remote = local, remote
    dual._remote_ids = {str(project.id)}
    monkeypatch.setattr(
        local, "scan_layer_source", lambda *_args, **_kwargs: pytest.fail("Scanned server files locally")
    )
    result = dual.scan_layer_source(binding.image_directory, maximum_frames=999, project_id=project.id)
    assert result.ready and len(result.jpg_files) == 2
    assert calls == [("POST", f"/api/v1/projects/{project.id}/layers/scan")]


@pytest.mark.parametrize("source_location", ["local", "server"])
def test_layer_link_dialog_defers_server_path_validation(qtbot, monkeypatch, source_location):
    dialog = LayerCreationDialog(maximum_frames=100, scanner=lambda *_args: None, source_location=source_location)
    qtbot.addWidget(dialog)
    dialog.name_edit.setText("Metal")
    dialog.tabs.setCurrentWidget(dialog.manual_tab)
    dialog.manual_images.edit.setText("server-only-images")
    calls = []

    def is_dir(path):
        calls.append(str(path))
        return False

    monkeypatch.setattr(Path, "is_dir", is_dir)
    dialog.ok_button.click()
    if source_location == "server":
        assert not calls
        assert dialog.result() == dialog.DialogCode.Accepted
    else:
        assert calls == ["server-only-images"]
        assert dialog.result() == dialog.DialogCode.Rejected
        assert dialog.validation_label.text()
