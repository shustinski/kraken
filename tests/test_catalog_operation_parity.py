"""Exercise operations through both catalog contracts and real application handlers."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService
from kraken_manager.domain.project import RepresentationKind
from kraken_server.app import create_app
from kraken_server.services import CommandContext, InMemoryServerServices, ValidationError
from test_server_workspace_inputs_parts.server_layer import _server_layer


def import_setup(tmp_path):
    server, local, session, project, layer, _binding, _representation, _context = _server_layer(tmp_path)
    server.profiles, server.identities = local.profiles, local.identities
    server.uow_factory = local._uow(str(project.id))
    server._pipeline_events = lambda identifier: iter(local.history(identifier))
    layer = next(item for item in local.list_layers(project.id) if item.id == layer.id)
    representation = local.create_representation(
        principal=session.principal,
        project=project,
        layer=layer,
        name="Imported",
        kind=RepresentationKind.IMAGE,
        idempotency_key="imported",
    )
    context = CommandContext(str(session.principal.id), "import", None)
    root = tmp_path / "server-input"
    root.mkdir()
    (root / "1_1.jpg").write_bytes(b"image-one")
    (root / "2_1.jpg").write_bytes(b"image-two")
    return server, local, session, project, layer, representation, context, root


@pytest.mark.parametrize("location", ["local", "server"])
def test_representation_import_uses_common_handlers_and_is_replayable(tmp_path, location):
    server, local, session, project, layer, representation, context, root = import_setup(tmp_path)
    if location == "local":
        service = local
    else:
        service = object.__new__(RemoteServerProjectService)

        def call(method, path, **kwargs):
            assert method == "POST"
            if path.endswith("/import-plan"):
                return server.plan_import_directory(str(project.id), kwargs["payload"], context)
            assert path.endswith(f"/representations/{representation.id}/import")
            return server.commit_managed_import(
                str(project.id), str(layer.id), str(representation.id), kwargs["payload"], context
            )

        service._call = call
    plan = service.plan_import_directory(project=project, directory=root)
    assert plan.ready
    arguments = dict(
        principal=session.principal,
        project=project,
        layer=layer,
        representation=representation,
        plan=plan,
        idempotency_key=context.idempotency_key,
    )
    result = service.commit_managed_import(**arguments)
    assert len(result.versions) == 2
    assert {(item.parameters["x"], item.parameters["y"]) for item in result.versions} == {(1, 1), (2, 1)}
    for version in result.versions:
        assert local.read_project_blob(project.id, version.sha256) == (root / version.filename).read_bytes()
    history = local.history(project.id)
    assert service.commit_managed_import(**arguments).versions == result.versions
    assert local.history(project.id) == history


def test_server_import_detects_file_changes_after_planning(tmp_path):
    server, _local, _session, project, layer, representation, context, root = import_setup(tmp_path)
    plan = server.plan_import_directory(str(project.id), {"directory": str(root)}, context)
    (root / "1_1.jpg").write_bytes(b"changed")
    with pytest.raises(ValidationError, match="SHA|sha|digest"):
        server.commit_managed_import(str(project.id), str(layer.id), str(representation.id), {"plan": plan}, context)


def test_history_filters_and_frame_bounds_match_local_contract():
    service = object.__new__(RemoteServerProjectService)
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    service._call = lambda *_args, **_kwargs: {
        "items": [
            {"event_id": "before", "event_type": "Wanted", "recorded_at": cutoff.isoformat()},
            {"event_id": "other", "event_type": "Other", "recorded_at": cutoff.isoformat()},
            {"event_id": "after", "event_type": "Wanted", "recorded_at": (cutoff + timedelta(seconds=1)).isoformat()},
        ]
    }
    assert [item.event_id for item in service.history("project", as_of=cutoff, event_types=("Wanted",))] == ["before"]
    service._call = lambda *_args, **_kwargs: {
        "items": [
            {"x": 1, "y": 1},
            {"x": 2, "y": 2},
            {"x": 3, "y": 3},
        ]
    }
    assert [(item.x, item.y) for item in service.frame_cells("p", "l", "r", bounds=(2, 2, 2, 2))] == [(2, 2)]


@pytest.mark.parametrize("use_digest", [True, False])
def test_remote_blob_reads_accept_local_digest_and_legacy_version_keys(use_digest):
    import hashlib

    content = b"published-result"
    digest = hashlib.sha256(content).hexdigest()
    version = SimpleNamespace(id="version", blob=SimpleNamespace(sha256=digest, size_bytes=len(content)))
    service = object.__new__(RemoteServerProjectService)
    service.auth = SimpleNamespace(access_token="token")
    service._http = SimpleNamespace(iter_bytes=lambda *_args, **_kwargs: (content,))
    service.list_artifact_series = lambda *_args, **_kwargs: (SimpleNamespace(id="series"),)
    service.artifact_versions = lambda *_args: (version,)
    service.artifact_version = lambda _project, key: (
        version if key == "version" else pytest.fail("Used a digest as a version ID")
    )
    assert service.read_project_blob("project", sha256=digest if use_digest else "version") == content


def test_audited_shared_operation_signatures_match():
    import inspect
    import json
    from pathlib import Path
    from kraken_hub.composition import EmbeddedProjectService

    inventory = json.loads(
        (Path(__file__).parents[1] / "docs/audits/2026-10-29-catalog-operations.json").read_text(encoding="utf-8")
    )
    for name in inventory["shared"]:
        local = inspect.signature(getattr(EmbeddedProjectService, name))
        remote = inspect.signature(getattr(RemoteServerProjectService, name))
        assert set(local.parameters) == set(remote.parameters), name


def test_positional_project_reads_route_to_remote_catalog(tmp_path):
    server, local, _session, project, _layer, representation, _context, _root = import_setup(tmp_path)
    remote = object.__new__(RemoteServerProjectService)
    calls = []
    remote.get_artifact_version = lambda *args: calls.append(args) or "remote-version"
    dual = object.__new__(DualCatalogService)
    dual.local, dual.remote, dual._remote_ids = local, remote, {str(project.id)}
    assert dual.get_artifact_version(project.id, representation.id) == "remote-version"
    assert calls == [(project.id, representation.id)]


def test_retry_ui_uses_server_queue_without_starting_local_agent():
    from kraken_hub.presentation.qt.parts.gateway_job import GatewayJobMixin

    controller = GatewayJobMixin()
    calls = []
    controller.service = SimpleNamespace(
        is_remote_project=lambda _id: True, retry_plugin_job=lambda **kwargs: calls.append(kwargs)
    )
    controller._gateway_for_job = lambda _job: pytest.fail("Started local gateway for a server retry")
    controller.session = SimpleNamespace(principal="actor")
    controller.my_work_refresh = None
    controller.retry_agent_job(SimpleNamespace(project_id="project"))
    assert calls[0]["gateway"] is None


def test_import_and_retry_routes_authenticate_and_validate_commands():
    calls = []

    class Services(InMemoryServerServices):
        def plan_import_directory(self, *args):
            calls.append(args)
            return {"items": []}

        def commit_managed_import(self, *args):
            calls.append(args)
            return {"versions": []}

        def retry_plugin_job(self, *args):
            calls.append(args)
            return {"state": "queued"}

    client = TestClient(create_app(services=Services(), development=True))
    headers = {"Authorization": "Bearer developer", "Idempotency-Key": "operation", "If-Match": "7"}
    for url, body in (
        ("/api/v1/projects/p/import-plan", {"directory": "server-source"}),
        ("/api/v1/projects/p/layers/l/representations/r/import", {"plan": {}}),
        ("/api/v1/projects/p/jobs/j/retry", {}),
    ):
        assert client.post(url, json=body).status_code == 401
        assert client.post(url, json=body, headers=headers).status_code == 200
    assert calls[-1][-1].expected_revision == 7


def test_temporal_catalog_reads_use_persistent_snapshots(tmp_path):
    server, local, session, project, layer, representation, _context, _root = import_setup(tmp_path)
    before = datetime.now(UTC)
    local.rename_project(
        principal=session.principal, project=local.get_project(project.id), name="New name", idempotency_key="rename"
    )
    assert server.get_project(str(project.id), as_of=before)["name"] == project.name
    assert server.get_project(str(project.id))["name"] == "New name"
    assert any(item["layer_id"] == str(layer.id) for item in server.list_layers(str(project.id), as_of=before))
    assert any(
        item["representation_id"] == str(representation.id)
        for item in server.list_representations(str(project.id), str(layer.id), as_of=before)
    )


def test_server_retry_executes_shared_application_command():
    from test_plugin_job_application import Fixture
    from kraken_manager.application.dto import CommandContext as ApplicationContext, SynchronizePluginJobCommand
    from kraken_manager.application.plugin_jobs import SynchronizePluginJobHandler
    from kraken_server.persistent_services import PostgresServerServices

    fixture = Fixture(shared=True)
    job = fixture.submit()
    recovered = SynchronizePluginJobHandler(
        fixture.factory, fixture.profiles, SimpleNamespace(now=lambda: datetime.now(UTC))
    )(
        SynchronizePluginJobCommand(
            context=ApplicationContext(actor=fixture.actor, idempotency_key="lost", gitlab_identity_verified=True),
            project_id=job.project_id,
            job_id=job.id,
            expected_revision=job.revision,
            state="recovery_required",
            error="agent lost",
        )
    )
    server = object.__new__(PostgresServerServices)
    server.uow_factory, server.profiles, server.agent_gateway = fixture.factory, fixture.profiles, fixture.gateway
    server._actor = lambda _id: fixture.actor
    result = server.retry_plugin_job(
        str(job.project_id), str(job.id), CommandContext(str(fixture.actor.id), "retry", recovered.revision)
    )
    assert result["state"] == "queued"
    assert len(fixture.gateway.submitted) == 2
    assert fixture.gateway.submitted[0] == fixture.gateway.submitted[1]


def test_server_project_template_is_created_and_replayable(tmp_path, monkeypatch):
    from kraken_manager.application.use_cases import CreateLayerHandler
    from kraken_manager.domain.project import LayerType

    server, local, session, project, layer, _representation, context, _root = import_setup(tmp_path)
    server.profiles = SimpleNamespace(profile=local.profile, get=local.profiles.get)
    layer = next(item for item in local.list_layers(project.id) if item.id == layer.id)
    local.rename_layer(
        principal=session.principal,
        project=local.get_project(project.id),
        layer=layer,
        name="Custom",
        idempotency_key="rename-initial-layer",
    )
    server._create_project = lambda _command: local.get_project(project.id)
    server._create_layer = CreateLayerHandler(local._uow(str(project.id)), local.profiles, local.clock)
    monkeypatch.setattr(server, "_ensure_server_workspace", lambda _project: None)
    payload = dict(
        name=project.name,
        width=project.width,
        height=project.height,
        storage_profile_id=project.storage_profile,
        layer_template=True,
    )
    result = server.create_project(payload, context)
    layers = local.list_layers(project.id)
    assert {item.name for item in layers}.issuperset(kind.value.capitalize() for kind in LayerType)
    history = local.history(project.id)
    assert server.create_project(payload, context) == result
    assert local.history(project.id) == history


@pytest.mark.parametrize("location", ["local", "server"])
def test_workspace_rename_uses_same_policy_and_storage_port(tmp_path, location):
    from pathlib import Path
    from dataclasses import replace
    from kraken_manager.application.project_commands import ProjectCommandService
    from kraken_manager.infrastructure.workspace_rename import FilesystemWorkspaceRenameStorage
    from kraken_manager.workspace import LayerSourceMode

    server, local, session, project, layer, _representation, context, _root = import_setup(tmp_path)
    registry = local.workspace_files.registry
    workspace = registry.get_project(str(project.id))
    binding = registry.get_layer(str(project.id), str(layer.id))
    image_directory = Path(workspace.source_project_dir) / "img" / layer.name
    image_directory.mkdir(parents=True)
    (image_directory / "image.jpg").write_bytes(b"image")
    registry.save_layer(
        str(project.id), replace(binding, mode=LayerSourceMode.MANAGED_COPY, image_directory=str(image_directory))
    )
    server.commands = ProjectCommandService(
        server.uow_factory,
        local.profiles,
        local.clock,
        rename_storage=FilesystemWorkspaceRenameStorage(local.workspace_files),
    )
    server._rename_layer = server.commands.execute
    server._rename_project = server.commands.execute
    layer = next(item for item in local.list_layers(project.id) if item.id == layer.id)
    if location == "local":
        renamed = local.rename_layer(
            principal=session.principal, project=project, layer=layer, name="Renamed", idempotency_key="rename-layer"
        )
        assert renamed.name == "Renamed"
    else:
        renamed = server.rename_layer(
            str(project.id), str(layer.id), "Renamed", CommandContext(context.actor_id, "rename-layer", layer.revision)
        )
        assert renamed["name"] == "Renamed"
    changed = registry.get_layer(str(project.id), str(layer.id))
    assert Path(changed.image_directory, "image.jpg").read_bytes() == b"image"
    assert not image_directory.exists()
    latest = local.get_project(project.id)
    if location == "local":
        local.rename_project(
            principal=session.principal, project=latest, name="Renamed project", idempotency_key="rename-project"
        )
    else:
        server.rename_project(
            str(project.id), "Renamed project", CommandContext(context.actor_id, "rename-project", latest.revision)
        )
    moved = registry.get_layer(str(project.id), str(layer.id))
    assert Path(moved.image_directory, "image.jpg").read_bytes() == b"image"
    assert local.get_project(project.id).name == "Renamed project"


def test_managed_import_accepts_an_opaque_nonfilesystem_source(tmp_path):
    from kraken_manager.application.imports import ImportPlan, ImportPlanItem, ImportSource, ImportContentMetadata
    from kraken_manager.application.managed_import import commit_managed_import

    server, local, session, project, layer, representation, _context, _root = import_setup(tmp_path)
    content = b"virtual image"
    source = ImportSource("memory://frame", "virtual.jpg", len(content))
    plan = ImportPlan(
        project.width, project.height, (ImportPlanItem(source, 1, 1),), (), len(content), project.frame_count - 1
    )
    calls = []

    class MemoryReader:
        def describe(self, item):
            calls.append(item.source_key)
            return ImportContentMetadata("virtual.jpg", "image/jpeg")

        def iter_bytes(self, item):
            assert item == source
            yield content

    versions = commit_managed_import(
        principal=session.principal,
        project=project,
        layer=layer,
        representation=representation,
        plan=plan,
        idempotency_key="virtual",
        projections=server.projections,
        events=server.events,
        uow_factory=server.uow_factory,
        profiles=local.profiles,
        clock=local.clock,
        content_reader=MemoryReader(),
    )
    assert calls == ["memory://frame"]
    assert local.read_project_blob(project.id, versions[0].sha256) == content


def test_temporal_frames_use_the_same_query_through_http_and_locally(tmp_path):
    from datetime import UTC, datetime
    from kraken_manager.application.results import FrameCellSnapshot

    server, local, session, project, layer, representation, context, root = import_setup(tmp_path)
    plan = server.plan_import_directory(str(project.id), {"directory": str(root)}, context)
    server.commit_managed_import(str(project.id), str(layer.id), str(representation.id), {"plan": plan}, context)
    before = datetime.now(UTC)
    old_cells = local.frame_cells(project.id, layer.id, representation.id, as_of=before)
    (root / "1_1.jpg").write_bytes(b"new content")
    updated_plan = server.plan_import_directory(str(project.id), {"directory": str(root)}, context)
    server.commit_managed_import(
        str(project.id),
        str(layer.id),
        str(representation.id),
        {"plan": updated_plan},
        CommandContext(context.actor_id, "updated-import", None),
    )

    class Services(InMemoryServerServices):
        def get_project(self, project_id, **kwargs):
            return server.get_project(project_id, **kwargs)

        def frame_cells(self, *args, **kwargs):
            return server.frame_cells(*args, **kwargs)

    client = TestClient(create_app(services=Services(), development=True))
    remote = object.__new__(RemoteServerProjectService)

    def call(method, path, **_kwargs):
        response = client.request(method, path, headers={"Authorization": "Bearer developer"})
        assert response.status_code == 200, response.text
        return response.json()

    remote._call = call
    historical = remote.frame_cells(project.id, layer.id, representation.id, as_of=before)
    assert historical == old_cells
    assert all(isinstance(item, FrameCellSnapshot) for item in historical)
    current = remote.frame_cells(project.id, layer.id, representation.id, bounds=(1, 1, 1, 1))
    assert len(current) == 1
    assert current[0].artifact_version_id != old_cells[0].artifact_version_id
