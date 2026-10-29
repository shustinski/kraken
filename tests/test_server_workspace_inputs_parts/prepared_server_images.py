"""test_server_workspace_inputs: test prepared server images."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from kraken_hub.manager_app import DesktopController
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteHttpClient, RemoteServerProjectService
from kraken_manager.application.plugin_jobs import SubmitPluginJobHandler
from kraken_manager.domain.project import RepresentationKind
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.services import CommandContext

from test_server_workspace_inputs_parts.server_layer import _server_layer

def test_remote_source_preparation_calls_server_and_returns_frames(tmp_path):
    _server, _local, session, project, layer, _binding, representation, _context = _server_layer(tmp_path)
    calls = []

    class Http(RemoteHttpClient):
        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
            return {"prepared_frames": 2} if method == "POST" else {"items": [{"x": 1, "y": 1}]}

    remote = RemoteServerProjectService(
        "http://example.test",
        auth=RemoteAuth("token", session.principal),
        data_dir=tmp_path / "remote",
        http=Http("http://example.test"),
    )
    cells = remote.materialize_representation_inputs(
        principal=session.principal,
        project=project,
        layer=layer,
        representation=representation,
    )
    assert cells[0].x == cells[0].y == 1
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith(f"/{representation.id}/prepare-inputs")
    assert calls[0][2]["timeout"] == 600

def test_selecting_linked_server_images_starts_preview_preparation(tmp_path, qapp):
    from kraken_manager.presentation.qt import ProjectManagerShell

    _server, _local, _session, project, layer, _binding, representation, _context = _server_layer(tmp_path)
    controller = object.__new__(DesktopController)
    controller._project_id = str(project.id)
    controller.service = SimpleNamespace(
        list_representations=lambda *_args: (representation,),
        is_remote_project=lambda _id: True,
    )
    controller._has_permission = lambda _permission: True
    refreshed = []
    controller._refresh_matrix = lambda *_args: refreshed.append(True)
    prepared = []
    controller._ensure_agent_source_ready = lambda **kwargs: prepared.append(kwargs)
    shell = ProjectManagerShell()
    workspace = shell.open_project_workspace()
    controller._load_representations(workspace, str(project.id), str(layer.id))
    assert workspace.image_representation_combo.currentData() == str(representation.id)
    assert len(prepared) == 1
    assert prepared[0]["source_representation_id"] == str(representation.id)
    prepared[0]["retry"]()
    assert len(refreshed) == 2
    workspace.matrix_view.close()
    shell.close()

def test_source_preparation_api_requires_session_and_passes_registered_ids():
    calls = []

    class Backend:
        def materialize_representation_inputs(self, *args):
            calls.append(args)
            return {"prepared_frames": 2}

    actor = SessionPrincipal("actor", "local", "token")
    client = TestClient(
        create_app(
            services=Backend(),
            session_resolver=lambda token: actor if token == "token" else None,
            project_access_mode="trusted_network",
        )
    )
    route = "/api/v1/projects/project/layers/layer/representations/source/prepare-inputs"
    assert client.post(route).status_code == 401
    response = client.post(route, headers={"Authorization": "Bearer token", "Idempotency-Key": "prepare"})
    assert response.status_code == 200
    assert response.json() == {"prepared_frames": 2}
    assert calls[0][:3] == ("project", "layer", "source")
    assert calls[0][3].idempotency_key == "prepare"

@pytest.mark.parametrize("action", ["prepare_dataset", "vectorize"])
def test_server_contour_actions_require_the_interactive_client_plugin(action):
    controller = object.__new__(DesktopController)
    controller.service = SimpleNamespace(is_remote_project=lambda _id: True)
    controller._project_id = "server"
    controller._has_permission = lambda _permission: True
    controller.plugin_items = {}
    assert controller._action_availability(action)[0] is False
    capability = controller._action_requirement(action)[1]
    controller.plugin_items = {
        "contour": SimpleNamespace(
            installed=True,
            metadata=SimpleNamespace(
                enabled=True,
                capabilities=(SimpleNamespace(operation=capability, modes=("interactive",)),),
            ),
        )
    }
    assert controller._action_availability(action) == (True, "")
    controller._has_permission = lambda _permission: False
    assert controller._action_availability(action)[0] is False

@pytest.mark.parametrize("action", ["prepare_dataset", "recognize_external", "vectorize"])
def test_pipeline_actions_accept_source_plugins_without_installed_distribution(action):
    from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory

    root = Path(__file__).resolve().parents[2]
    plugin_id = "neuralimage" if action == "recognize_external" else "contour"
    plugin = load_plugin_metadata_from_directory(root / "plugins" / plugin_id)
    controller = object.__new__(DesktopController)
    controller.service = SimpleNamespace(is_remote_project=lambda _id: True)
    controller._project_id = "server"
    controller._has_permission = lambda _permission: True
    controller.plugin_items = {plugin_id: PluginInventoryItem(plugin, installed=False)}
    assert controller._action_availability(action) == (True, "")
    from dataclasses import replace

    controller.plugin_items[plugin_id] = PluginInventoryItem(
        replace(plugin, source_dir=str(root / "missing-plugin")), installed=False
    )
    assert controller._action_availability(action)[0] is False

@pytest.mark.parametrize("capability", ["frames.vectorize.v1", "frames.dataset.prepare.v1"])
@pytest.mark.parametrize("input_transport", ["files", "server"])
def test_prepared_server_images_can_be_submitted_to_contour_queue(tmp_path, capability, input_transport):
    server, local, session, project, layer, _binding, representation, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
    target = local.create_representation(
        principal=session.principal,
        project=project,
        layer=layer,
        name="Vectors",
        kind=RepresentationKind.VECTOR,
        source_image_representation_id=representation.id,
        idempotency_key="target",
    )
    manifests = []
    gateway = SimpleNamespace(is_available=lambda *_args: True, submit=manifests.append)
    server._submit_plugin_job = SubmitPluginJobHandler(
        local._uow(str(project.id)),
        local.profiles,
        local.clock,
        gateway,
    )
    job = server.submit_plugin_job(
        str(project.id),
        {
            "layer_id": str(layer.id),
            "source_representation_id": str(representation.id),
            "target_representation_id": str(target.id),
            "coordinates": [[1, 1], [1, 2]],
            "capability": capability,
            "parameters": {"input_transport": input_transport},
        },
        CommandContext(str(session.principal.id), "submit", None),
    )
    assert job["capability"] == capability
    assert len(manifests) == 1
    assert len(manifests[0].inputs) == 2
    assert all(local.get_artifact_version(project.id, item.artifact_version_id) for item in manifests[0].inputs)
    assert manifests[0].parameters["project_width"] == project.width
    for item in manifests[0].inputs:
        version = local.get_artifact_version(project.id, item.artifact_version_id)
        assert manifests[0].parameters["input_media_types"][str(version.id)] == version.media_type
        if input_transport == "server":
            assert Path(item.relative_path).name == version.filename
