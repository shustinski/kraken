from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PyQt6.QtGui import QImage

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.manager_app import DesktopController
from kraken_hub.matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource
from kraken_hub.remote_client import RemoteAuth, RemoteHttpClient, RemoteServerProjectService
from kraken_core.frame_matrix import MatrixBounds, MatrixViewportRequest
from kraken_manager.application.use_cases import AddArtifactVersionHandler, CreateArtifactSeriesHandler
from kraken_manager.application.plugin_jobs import SubmitPluginJobHandler
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.persistent_services import PostgresServerServices
from kraken_server.services import CommandContext, ForbiddenError, NotFoundError, ValidationError


def test_persistent_history_serializes_nested_event_payloads(tmp_path):
    from kraken_server.services import InMemoryServerServices

    server, local, _session, project, *_rest = _server_layer(tmp_path)
    events = local.history(project.id)
    server.events = SimpleNamespace(list_project_events=lambda _project_id, **_kwargs: tuple(enumerate(events, 1)))

    class HistoryServices(InMemoryServerServices):
        def history(self, project_id, *, cursor, limit):
            return server.history(str(project.id), cursor=cursor, limit=limit)

    client = TestClient(create_app(services=HistoryServices(), development=True))
    headers = {"Authorization": "Bearer developer"}
    created = client.post(
        "/api/v1/projects",
        headers={**headers, "Idempotency-Key": "history-project"},
        json={"name": "History", "width": 2, "height": 2, "orientation": "y_down"},
    )
    response = client.get(
        f"/api/v1/projects/{created.json()['project_id']}/history",
        headers=headers,
        params={"limit": 500},
    )
    assert response.status_code == 200
    from kraken_manager.infrastructure.filesystem._codec import encode_model

    assert [item["payload"] for item in response.json()["items"]] == [encode_model(event.payload) for event in events]


def _server_layer(tmp_path: Path, orientation=GridOrientation.Y_DOWN, *, images: Path | None = None, width=2, height=2):
    if images is None:
        images = tmp_path / "linked-images"
        images.mkdir()
        for number in (0, 2):
            Image.new("RGB", (16, 12), "red").save(images / f"{number:04d}.jpg")
    local = EmbeddedProjectService(tmp_path / "data")
    session = local.create_initial_account("owner", "Owner", "")
    project = local.create_project(
        principal=session.principal,
        name="Server layer",
        width=width,
        height=height,
        orientation=orientation,
        idempotency_key="project",
    )
    layer, binding, representation = local.create_external_layer(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=1,
        image_directory=images,
        ssc_directory=None,
        prv_directory=None,
        idempotency_key="layer",
    )
    server = object.__new__(PostgresServerServices)
    server.projections = local._projection(project.id)
    server.workspace_files = local.workspace_files
    server.events = FilesystemEventStore(local.catalog_root, str(project.id))
    server._actor = lambda _id: session.principal
    server._create_artifact_series = CreateArtifactSeriesHandler(
        local._uow(str(project.id)), local.profiles, local.clock
    )
    server._add_artifact_version = AddArtifactVersionHandler(local._uow(str(project.id)), local.profiles, local.clock)
    context = CommandContext(str(session.principal.id), "prepare", None)
    return server, local, session, project, layer, binding, representation, context


@pytest.mark.parametrize("orientation", list(GridOrientation))
def test_server_linked_source_has_previews_and_stable_inputs(tmp_path, orientation):
    server, local, _session, project, layer, binding, representation, context = _server_layer(tmp_path, orientation)
    result = server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
    assert result == {"prepared_frames": 2}
    viewport = server.matrix_viewport(
        str(project.id),
        layer_id=str(layer.id),
        representation_ids=(str(representation.id),),
        x1=1,
        y1=1,
        x2=2,
        y2=2,
        lod=0,
        include_missing=False,
    )
    cells = viewport["cells"]
    assert {(cell["x"], cell["y"]) for cell in cells} == {(1, 1), (1, 2)}
    for cell in cells:
        version = local.get_artifact_version(project.id, cell["artifact_version_id"])
        assert version is not None
        assert cell["asset_sha256"] == version.sha256
        original = Path(binding.image_directory) / version.filename
        assert local.read_project_blob(project.id, version.sha256) == original.read_bytes()
    history = local.history(project.id)
    assert (
        server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
        == result
    )
    assert local.history(project.id) == history
    assert len(tuple(Path(binding.image_directory).iterdir())) == 2


def test_server_source_preparation_resumes_after_missing_file(tmp_path):
    server, local, _session, project, layer, binding, representation, context = _server_layer(tmp_path)
    missing = Path(binding.image_directory) / "0002.jpg"
    contents = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValidationError, match="исходные изображения"):
        server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
    missing.write_bytes(contents)
    assert server.materialize_representation_inputs(
        str(project.id),
        str(layer.id),
        str(representation.id),
        context,
    ) == {"prepared_frames": 2}
    assert len(local.frame_cells(project.id, layer.id, representation.id)) == 2


def test_server_source_preparation_rejects_wrong_project(tmp_path):
    server, _local, _session, project, layer, _binding, representation, context = _server_layer(tmp_path)
    from uuid import uuid4

    with pytest.raises(NotFoundError):
        server.materialize_representation_inputs(str(uuid4()), str(layer.id), str(representation.id), context)
    assert server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)


def test_server_source_preparation_preserves_import_permissions(tmp_path):
    from kraken_manager.domain.identity import Principal

    server, local, _session, project, layer, _binding, representation, context = _server_layer(tmp_path)
    history = local.history(project.id)
    server._actor = lambda _id: Principal.local(subject="reader", display_name="Reader")
    with pytest.raises(ForbiddenError):
        server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
    assert local.history(project.id) == history
    assert local.frame_cells(project.id, layer.id, representation.id) == ()


def test_server_preview_uses_version_id_and_decodes_linked_image(tmp_path):
    server, local, _session, project, layer, _binding, representation, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)

    class MatrixService:
        matrix_viewport = server.matrix_viewport

        def read_project_blob(self, project_id, version_id):
            version = local.get_artifact_version(project_id, version_id)
            assert version is not None
            return local.read_project_blob(project_id, version.sha256)

    service = MatrixService()
    source = KrakenMatrixDataSource(
        service,
        project_id=str(project.id),
        layer_id=str(layer.id),
        representation_ids=(str(representation.id),),
        matrix_width=2,
    )
    item = source.load_viewport(MatrixViewportRequest(MatrixBounds(1, 1, 2, 2))).items[0]
    assert item.asset is not None
    assert item.asset.source_key == item.metadata["artifact_version_id"]
    assert item.asset.source_key != item.metadata["asset_sha256"]
    output = KrakenMatrixAssetSource(service, project_id=str(project.id)).load_asset(item.asset, width=32, height=24)
    image = QImage.fromData(output)
    assert not image.isNull()
    assert (image.width(), image.height()) == (32, 24)


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

    root = Path(__file__).resolve().parents[1]
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
def test_prepared_server_images_can_be_submitted_to_contour_queue(tmp_path, capability):
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
            "parameters": {},
        },
        CommandContext(str(session.principal.id), "submit", None),
    )
    assert job["capability"] == capability
    assert len(manifests) == 1
    assert len(manifests[0].inputs) == 2
    assert all(local.get_artifact_version(project.id, item.artifact_version_id) for item in manifests[0].inputs)


@pytest.mark.parametrize("capability", ["frames.vectorize.v1", "frames.binary-segment.v2"])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("existing_target", ["missing", "linked", "unrelated"])
def test_contour_launch_creates_or_reuses_empty_output(tmp_path, monkeypatch, remote, existing_target, capability):
    from kraken_hub import manager_app
    from kraken_manager.domain.project import RepresentationPurpose

    output_kind = RepresentationKind.VECTOR if capability == "frames.vectorize.v1" else RepresentationKind.IMAGE
    output_purpose = (
        RepresentationPurpose.VECTOR if capability == "frames.vectorize.v1" else RepresentationPurpose.BINARY
    )

    server, local, session, project, layer, _binding, source, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(source.id), context)
    target = None
    if existing_target != "missing":
        target_source = source
        if existing_target == "unrelated":
            target_source = local.create_representation(
                principal=session.principal,
                project=project,
                layer=layer,
                name="Other images",
                kind=RepresentationKind.IMAGE,
                idempotency_key="other-source",
            )
        target = local.create_representation(
            principal=session.principal,
            project=project,
            layer=layer,
            name="Existing CIF",
            kind=output_kind,
            purpose=output_purpose,
            source_image_representation_id=target_source.id,
            idempotency_key="existing-output",
        )
    manifests = []
    gateway = SimpleNamespace(is_available=lambda *_args: True, submit=manifests.append)
    monkeypatch.setattr(manager_app, "AgentPluginGateway", lambda **_kwargs: gateway)
    local.is_remote_project = lambda _id: remote
    submit = local.submit_plugin_job
    local.submit_plugin_job = lambda **kwargs: submit(**{**kwargs, "gateway": gateway})
    controller = object.__new__(DesktopController)
    controller.service = local
    controller.session = session
    controller._project_id = str(project.id)
    controller._workspace = SimpleNamespace(
        matrix_view=SimpleNamespace(selected_coordinates=lambda **_kwargs: ()),
        vector_representation_combo=SimpleNamespace(currentData=lambda: str(target.id) if target is not None else None),
    )
    controller.agent_runtime = SimpleNamespace(
        interactive_registry=lambda _operation: "interactive-registry",
        execution_registry=lambda _operation: "execution-registry",
        ensure_started=lambda: frozenset({capability}),
        base_url="http://localhost:9999",
        token="test",
        protocol_by_capability={},
    )
    launched = []
    controller._run_remote_interactive_job = lambda job, **kwargs: launched.append((job, kwargs))
    job = controller._submit_agent_action(
        layer_id=str(layer.id),
        source_representation_id=str(source.id),
        capability=capability,
        parameters={},
    )
    vectors = [
        item
        for item in local.list_representations(project.id, layer.id)
        if item.kind is output_kind
        and item.purpose is output_purpose
        and item.source_image_representation_id == source.id
    ]
    assert len(vectors) == 1
    assert vectors[0].source_image_representation_id == source.id
    assert job.target_representation_id == vectors[0].id
    if existing_target == "linked":
        assert vectors[0].id == target.id
    assert len(manifests) == 1
    assert len(manifests[0].inputs) == 2
    assert local.frame_cells(project.id, layer.id, vectors[0].id) == ()
    retry = controller._submit_agent_action(
        layer_id=str(layer.id),
        source_representation_id=str(source.id),
        capability=capability,
        parameters={},
    )
    assert retry.target_representation_id == job.target_representation_id
    assert len(manifests) == 2
    assert len(launched) == (2 if remote else 0)
    if remote:
        assert manifests[0].parameters["execution_host"] == "desktop"


def test_partial_server_inputs_are_completed_before_launch(qtbot):
    from PyQt6.QtWidgets import QMainWindow

    cells = [SimpleNamespace(x=1, y=1)]
    layer = SimpleNamespace(id="layer")
    representation = SimpleNamespace(id="source")

    def prepare(**_kwargs):
        cells.append(SimpleNamespace(x=1, y=2))
        return tuple(cells)

    controller = object.__new__(DesktopController)
    controller._project_id = "project"
    controller.session = SimpleNamespace(principal=object())
    controller.shell = QMainWindow()
    qtbot.addWidget(controller.shell)
    controller._background_workers = set()
    controller._source_preparation_keys = set()
    controller.service = SimpleNamespace(
        get_project=lambda _id: SimpleNamespace(id="project"),
        frame_cells=lambda *_args: tuple(cells),
        layer_file_binding=lambda *_args: SimpleNamespace(frame_positions={"0000.jpg": 1, "0002.jpg": 3}),
        list_layers=lambda _id: (layer,),
        list_representations=lambda *_args: (representation,),
        materialize_representation_inputs=prepare,
    )
    retried = []
    assert (
        controller._ensure_agent_source_ready(
            layer_id="layer",
            source_representation_id="source",
            retry=lambda: retried.append(True),
        )
        is False
    )
    qtbot.waitUntil(lambda: bool(retried), timeout=3000)
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=3000)
    assert len(cells) == 2


def _desktop_job(tmp_path, capability):
    import sqlalchemy as sa
    from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
    from kraken_server.agent_gateway import PostgresAgentGateway

    server, local, session, project, layer, _binding, source, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(source.id), context)
    target = (
        local.create_representation(
            principal=session.principal,
            project=project,
            layer=layer,
            name="Contour output",
            kind=RepresentationKind.VECTOR,
            source_image_representation_id=source.id,
            idempotency_key="target",
        )
        if capability == "frames.vectorize.v1"
        else source
    )
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'queue.sqlite3'}")
    blobs = FilesystemBlobStore.for_project(local.catalog_root, str(project.id))
    gateway = PostgresAgentGateway(engine, tokens=None, blobs=blobs)
    gateway.metadata.create_all(engine)
    server.uow_factory = local._uow(str(project.id))
    server.uow_factory.blobs = blobs
    server.profiles = local.profiles
    server.identities = local.identities
    server.project_ids_for_principal = lambda principal_id: (
        (str(project.id),) if local.project_roles(project.id, principal_id) else ()
    )
    server.agent_gateway = gateway
    server._submit_plugin_job = SubmitPluginJobHandler(server.uow_factory, local.profiles, local.clock, gateway)
    job = server.submit_plugin_job(
        str(project.id),
        {
            "layer_id": str(layer.id),
            "source_representation_id": str(source.id),
            "target_representation_id": str(target.id),
            "coordinates": [[1, 1], [1, 2]],
            "capability": capability,
            "parameters": {"execution_host": "desktop"},
        },
        CommandContext(str(session.principal.id), "desktop-submit", None),
    )
    return server, local, session, project, layer, target, gateway, job


def test_desktop_lease_is_limited_to_its_initiator_and_job(tmp_path):
    from uuid import uuid4
    from kraken_server.agent_auth import AgentIdentity

    server, _local, session, project, _layer, _target, gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    app = create_app(
        services=server,
        session_resolver=lambda token: SessionPrincipal(
            str(session.principal.id) if token == "owner" else str(uuid4()), "local", token
        ),
        agent_gateway=gateway,
    )
    client = TestClient(app)
    endpoint = f"/api/v1/projects/{project.id}/jobs/{job['id']}/desktop-lease"
    assert client.post(endpoint).status_code == 401
    assert client.post(endpoint, headers={"Authorization": "Bearer other"}).status_code == 403
    assert gateway.lease(AgentIdentity("machine", "server", frozenset({"frames.vectorize.v1"}))) is None
    headers = {"Authorization": "Bearer owner"}
    assert client.post(endpoint.replace(str(project.id), str(uuid4())), headers=headers).status_code == 404
    response = client.post(endpoint, headers=headers)
    assert response.status_code == 200
    assert response.json()["job"]["job_id"] == job["id"]
    assert client.post(endpoint, headers=headers).status_code == 409
    assert client.post("/api/v1/agent/lease", headers=headers, json={}).status_code == 401
    assert client.post(f"/api/v1/agent/jobs/{job['id']}/heartbeat", headers=headers, json={}).status_code == 200
    from kraken_manager.application.plugin_jobs import CancelPluginJobHandler

    server._cancel_plugin_job = CancelPluginJobHandler(server.uow_factory, server.profiles, _local.clock, gateway)
    current = _local.plugin_jobs()[0]
    assert current.state.value == "waiting_for_user"
    cancelled = client.post(
        f"/api/v1/projects/{project.id}/jobs/{job['id']}/cancel",
        json={},
        headers={**headers, "Idempotency-Key": "cancel", "If-Match": str(current.revision)},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert client.post(f"/api/v1/agent/jobs/{job['id']}/heartbeat", headers=headers, json={}).status_code == 403


def test_server_task_cancellation_does_not_start_local_agent():
    controller = object.__new__(DesktopController)
    calls = []
    controller.service = SimpleNamespace(
        is_remote_project=lambda _id: True, cancel_plugin_job=lambda **kwargs: calls.append(kwargs)
    )
    controller.session = SimpleNamespace(principal="owner")
    controller.my_work_refresh = None
    controller._gateway_for_job = lambda _job: pytest.fail("Server cancellation must not start the local Agent")
    job = SimpleNamespace(project_id="server")
    controller.cancel_agent_job(job)
    assert calls[0]["gateway"] is None
    assert calls[0]["job"] is job


def test_server_recognition_stages_model_with_its_original_extension(tmp_path):
    import hashlib
    from kraken_agent.jobs import StagingWorkspace
    from kraken_agent.server_worker import ServerAgentWorker
    from kraken_agent.runner import PluginRegistry
    from kraken_manager.infrastructure.filesystem._codec import decode_model
    from kraken_manager.domain.workflows import PluginJobManifestV1
    from neuralimage.kraken_bridge import HeadlessOptions, NeuralImageKrakenSession

    _server, _local, _session, _project, _layer, _target, gateway, payload = _desktop_job(
        tmp_path, "frames.dataset.prepare.v1"
    )
    from kraken_server.agent_auth import AgentIdentity

    lease = gateway.lease(
        AgentIdentity(f"desktop:{payload['actor_principal_id']}", "desktop", frozenset({"frames.dataset.prepare.v1"})),
        job_id=payload["id"],
    )
    raw = lease["manifest"]
    model = tmp_path / "cached-model-without-extension"
    model.write_bytes(b"test-weights")
    raw["capability"] = "frames.binary-segment.v2"
    raw["parameters"] = {
        "execution_host": "desktop",
        "model_source_path": str(model),
        "model_filename": "weights.pth",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    manifest = decode_model(PluginJobManifestV1, raw)
    workspace = StagingWorkspace(tmp_path / "staging", payload["id"])
    workspace.create()
    worker = ServerAgentWorker(
        server_url="http://localhost", token="owner", data_dir=tmp_path / "worker", registry=PluginRegistry({})
    )

    def download(_endpoint, destination, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), "red").save(destination)
        return "image/jpeg"

    worker._download = download
    transport = worker._transport_manifest(manifest, workspace)
    assert transport.parameters["model_relative_path"] == "inputs/model/weights.pth"
    assert (workspace.path / "inputs/model/weights.pth").read_bytes() == model.read_bytes()
    # The normal recognition bridge accepts the staged, pinned model.
    session = NeuralImageKrakenSession(
        transport, workspace.path, workspace.path / "result.json", (), (), workspace.path / "outputs"
    )
    options = HeadlessOptions.from_session(session)
    assert options.model_path.suffix == ".pth"


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


@pytest.mark.parametrize("filename", ["../escape.jpg", "/absolute.jpg", "images/a.jpg:stream"])
def test_neural_training_rejects_unsafe_dataset_paths(tmp_path, filename):
    import hashlib
    from zipfile import ZipFile
    from neuralimage.kraken_bridge import NeuralImageTrainSession

    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr(filename, b"unsafe")
    manifest = SimpleNamespace(
        inputs=(
            SimpleNamespace(
                relative_path="dataset.zip",
                media_type="application/zip",
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            ),
        )
    )
    session = NeuralImageTrainSession(manifest, tmp_path, tmp_path / "result.json")
    with pytest.raises(ValueError):
        session.prepare_dataset()
    assert not (tmp_path.parent / "escape.jpg").exists()


@pytest.mark.parametrize("protocol", ["v1", "v2"])
def test_training_window_receives_samples_and_returns_model(qapp, tmp_path, monkeypatch, protocol):
    import hashlib
    import sys
    from uuid import uuid4
    from zipfile import ZipFile
    from PyQt6.QtWidgets import QMainWindow, QMessageBox
    from kraken_core.plugin_protocol import PluginAsset, PluginFrameInput, PluginJobManifest, PluginJobManifestV2
    from neuralimage.kraken_bridge import NeuralImageTrainSession

    archive = tmp_path / "dataset.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("images/0000.jpg", b"sample")
        stream.writestr("cif/0000.cif", b"L M1; E;")
    identity = str(uuid4())
    input_values = dict(
        frame_id=identity,
        x=1,
        y=1,
        relative_path="dataset.zip",
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        media_type="application/zip",
    )
    common = dict(
        job_id=identity, project_id=identity, layer_id=identity, actor_id=identity, operation="dataset.model.train.v1"
    )
    if protocol == "v1":
        manifest = PluginJobManifest(
            **common,
            target_representation_id=identity,
            inputs=(PluginFrameInput(artifact_version_id=identity, **input_values),),
        )
    else:
        manifest = PluginJobManifestV2(
            **common, inputs=(PluginAsset(asset_id=identity, scope="frame", role="input", **input_values),)
        )
    session = NeuralImageTrainSession(manifest, tmp_path, tmp_path / "result.json")
    session.output_directory.mkdir()
    window = QMainWindow()
    values = {}
    window.set_jpg_path = lambda value: values.update(images=value)
    window.set_label_path = lambda value: values.update(cif=value)
    window.set_result_path = lambda value: values.update(result=value)
    state = SimpleNamespace()

    def execute():
        assert Path(values["images"]).is_dir()
        assert Path(values["cif"]).is_dir()
        assert state.sample_folder == values["images"]
        assert state.label_folder == values["cif"]
        assert state.result_folder == values["result"]
        (session.output_directory / "weights.pth").write_bytes(b"test-model")
        window._kraken_return_action.trigger()

    monkeypatch.setitem(
        sys.modules,
        "neuralimage.controller",
        SimpleNamespace(
            AppController=lambda **_kwargs: SimpleNamespace(
                main_window_presenter=SimpleNamespace(view=window, main_window_state=state), exec=execute
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: QMessageBox.StandardButton.Ok)
    session.run_interactive()
    assert session.result_manifest_path.is_file()


@pytest.mark.parametrize("fails", [False, True])
def test_desktop_handoff_reports_result_and_refreshes_on_qt_thread(qtbot, fails):
    from threading import get_ident
    from PyQt6.QtWidgets import QMainWindow

    controller = object.__new__(DesktopController)
    controller.shell = QMainWindow()
    qtbot.addWidget(controller.shell)
    controller._project_id = "project"
    controller._workspace = SimpleNamespace(_selected_layer_id="layer")
    controller.my_work_refresh = None
    controller._background_workers = set()
    calls = []
    refreshed = []
    errors = []
    qt_thread = get_ident()

    def execute(project_id, job_id, *, registry):
        calls.append((project_id, job_id, registry, get_ident()))
        if fails:
            raise RuntimeError("Contour could not publish the result")
        return {"imported": {"job": {"state": "succeeded"}}}

    controller.service = SimpleNamespace(run_interactive_job=execute)
    controller._load_representations = lambda *_args: refreshed.append(get_ident())
    controller._refresh_layer_pipeline = lambda *_args: refreshed.append(get_ident())
    controller._error = errors.append
    controller._run_remote_interactive_job(
        SimpleNamespace(project_id="project", layer_id="layer", id="job"), registry="registry"
    )
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=3000)
    assert len(calls) == 1
    assert calls[0][:3] == ("project", "job", "registry")
    assert calls[0][3] != qt_thread
    assert refreshed == ([] if fails else [qt_thread, qt_thread])
    assert bool(errors) is fails


@pytest.mark.parametrize("transfer_mode", ["proxy", "gateway"])
@pytest.mark.parametrize("capability", ["frames.vectorize.v1", "frames.dataset.prepare.v1"])
def test_interactive_contour_opens_with_images_and_returns_results(tmp_path, monkeypatch, capability, transfer_mode):
    import os
    import socket
    import sys
    import threading
    import time
    import uvicorn
    from kraken_agent.runner import PluginProcessSpec, PluginRegistry
    from kraken_agent.server_worker import ServerAgentWorker
    from kraken_manager.infrastructure.filesystem._codec import decode_model
    from kraken_manager.domain.workflows import PluginJob

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(root / "src"), str(root / "plugins/contour/src"), str(root / "plugins/neuralimage/src"))),
    )

    server, local, session, project, layer, target, gateway, payload = _desktop_job(tmp_path, capability)
    downloads = []
    input_tickets = []
    original_request = ServerAgentWorker._request

    def request_with_ticket_check(worker, method, path, payload=None):
        response = original_request(worker, method, path, payload)
        if path.endswith("/download-ticket"):
            version_id = path.split("/")[-2]
            metadata = server.get_artifact_version(str(project.id), version_id)
            assert response["media_type"] == metadata["media_type"]
            input_tickets.append(response)
        return response

    monkeypatch.setattr(ServerAgentWorker, "_request", request_with_ticket_check)

    def transfer_ticket(operation, _digest, _size, *, context):
        if operation != "download":
            return {"mode": "proxy"}
        return {"mode": "gateway", "url": f"{base_url}/test-blobs/{context['version']}", "token": "test-transfer"}

    blob_transfer = SimpleNamespace(ticket=transfer_ticket) if transfer_mode == "gateway" else None
    app = create_app(
        services=server,
        session_resolver=lambda token: (
            SessionPrincipal(str(session.principal.id), "local", token) if token == "owner" else None
        ),
        agent_gateway=gateway,
        blob_gateway=blob_transfer,
    )
    from fastapi.responses import StreamingResponse

    @app.get("/test-blobs/{version_id}")
    def raw_blob(version_id: str):
        downloads.append(version_id)
        return StreamingResponse(
            server.iter_artifact_bytes(str(project.id), version_id), media_type="application/octet-stream"
        )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    http_server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: http_server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not http_server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert http_server.started
    script = tmp_path / "contour_interactive.py"
    script.write_text(
        """
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import faulthandler
faulthandler.enable()
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
from pathlib import Path
from zipfile import ZipFile
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox
from contour.kraken_bridge import prepare_contour_launch
from contour.application import bootstrap
from contour.application.cli import main
session, arguments = prepare_contour_launch([])
assert len(session.input_paths) == 2
original_build = bootstrap.build_application
def build(argv):
    global app, window
    app, window = original_build([*argv, "--no-qss", "--log-file", str(session.staging_root / "contour.log")])
    QTimer.singleShot(1000, publish)
    return app, window
bootstrap.build_application = build
def publish():
    try:
        assert window.isVisible()
        assert window._widget._image_list_model.rowCount() == 2
        window.grab().save(str(session.staging_root / "contour-window.png"))
        if session.manifest.operation == "frames.vectorize.v1":
            for image in session.input_paths:
                (session.output_directory / (image.stem + ".cif")).write_text("L M1; P 0 0 10 0 10 10 0 10; E;\\n")
        else:
            import shutil
            (session.output_directory / "images").mkdir(exist_ok=True)
            (session.output_directory / "cif").mkdir(exist_ok=True)
            for image in session.input_paths:
                shutil.copyfile(image, session.output_directory / "images" / image.name)
                (session.output_directory / "cif" / (image.stem + ".cif")).write_text("L M1; E;")
        QMessageBox.information = lambda *args: QMessageBox.StandardButton.Ok
        window._kraken_return_action.trigger()
    finally:
        app.quit()
main([])
""",
        encoding="utf-8",
    )
    registry = PluginRegistry(
        {capability: PluginProcessSpec(capability, (sys.executable, str(script)), interactive=True)}
    )
    remote = RemoteServerProjectService(
        base_url, auth=RemoteAuth("owner", session.principal), data_dir=tmp_path / "desktop"
    )
    try:
        result = remote.run_interactive_job(project.id, payload["id"], registry=registry)
        assert result["imported"]["job"]["state"] == "succeeded"
        assert len(downloads) == (2 if transfer_mode == "gateway" else 0)
        assert len(input_tickets) == 2
        assert result["imported"]["requires_partial_confirmation"] is False
        job = decode_model(PluginJob, payload)
        cells = local.frame_cells(project.id, layer.id, target.id)
        assert len(cells) == 2
        versions = result["imported"]["versions"]
        assert len(versions) == (2 if capability == "frames.vectorize.v1" else 1)
        assert all(local.read_project_blob(project.id, version["sha256"]) for version in versions)
        if capability == "frames.dataset.prepare.v1":
            assert all(
                local.get_artifact_version(project.id, cell.artifact_version_id).media_type == "image/jpeg"
                for cell in cells
            )
            series = local._projection(project.id).get_artifact_series(versions[0]["series_id"])
            assert series.scope.value == "layer_attachment"
            assert len(versions[0]["input_version_ids"]) == 2
            training = server.submit_plugin_job(
                str(project.id),
                {
                    "layer_id": str(layer.id),
                    "source_representation_id": str(target.id),
                    "target_representation_id": str(target.id),
                    "coordinates": [[1, 1], [1, 2]],
                    "capability": "dataset.model.train.v1",
                    "parameters": {"execution_host": "desktop", "dataset_version_id": versions[0]["id"]},
                },
                CommandContext(str(session.principal.id), "train", None),
            )
            train_script = tmp_path / "training.py"
            train_script.write_text(
                """
import os
from neuralimage.kraken_bridge import load_session_from_values
session = load_session_from_values(job_manifest=None, result_manifest=None, staging_root=None)
images, labels = session.prepare_dataset()
assert len(list(images.glob('*.jpg'))) == 2
assert len(list(labels.glob('*.cif'))) == 2
(session.output_directory / 'weights.pth').write_bytes(b'test-model')
session.publish_model()
""",
                encoding="utf-8",
            )
            train_registry = PluginRegistry(
                {
                    "dataset.model.train.v1": PluginProcessSpec(
                        "dataset.model.train.v1", (sys.executable, str(train_script)), interactive=True
                    )
                }
            )
            trained = remote.run_interactive_job(project.id, training["id"], registry=train_registry)
            assert trained["imported"]["job"]["state"] == "succeeded"
            assert len(trained["imported"]["versions"]) == 1
            controller = object.__new__(DesktopController)
            controller.service = local
            snapshot = controller._pipeline_snapshot(project.id, layer.id)
            output_nodes = [node for lane in snapshot.lanes for node in lane.nodes if node.kind in {"dataset", "model"}]
            assert len(output_nodes) == 2
            assert all(node.details["artifact_version_ids"] for node in output_nodes)

            for model in trained["imported"]["versions"]:
                model_series = local._projection(project.id).get_artifact_series(model["series_id"])
                assert model_series.scope.value == "layer_attachment"
            assert all(
                local.get_artifact_version(project.id, cell.artifact_version_id).media_type == "image/jpeg"
                for cell in local.frame_cells(project.id, layer.id, target.id)
            )

        assert (
            remote.data_dir / "desktop-jobs" / str(job.id) / "staging" / str(job.id) / "contour-window.png"
        ).is_file()
    finally:
        http_server.should_exit = True
        thread.join(timeout=5)
        sock.close()
