"""test_server_workspace_inputs: server layer."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application.use_cases import AddArtifactVersionHandler, CreateArtifactSeriesHandler
from kraken_manager.domain.project import GridOrientation, LayerType
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_server.app import create_app
from kraken_server.persistent_services import PostgresServerServices
from kraken_server.services import CommandContext

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

def test_attached_layer_replay_reuses_storage_and_authorizes_existing_layer(tmp_path):
    from kraken_manager.application.use_cases import CreateLayerHandler, CreateRepresentationHandler

    server, local, session, project, layer, binding, representation, _context = _server_layer(tmp_path)
    server._create_layer = CreateLayerHandler(local._uow(str(project.id)), local.profiles, local.clock)
    server._create_representation = CreateRepresentationHandler(
        local._uow(str(project.id)), local.profiles, local.clock
    )
    context = CommandContext(str(session.principal.id), "layer", None)
    history = local.history(project.id)

    def forbidden_bind(*_args):
        pytest.fail("Replayed attachment must not create or copy another layer folder")

    result = server._attach_workspace_layer(
        str(project.id),
        {"name": layer.name, "type": layer.type.value, "order": layer.order},
        context,
        bind=forbidden_bind,
        managed=False,
    )
    assert result["layer"]["layer_id"] == str(layer.id)
    assert result["representation"]["representation_id"] == str(representation.id)
    assert local.history(project.id) == history
    assert Path(binding.image_directory).is_dir()
