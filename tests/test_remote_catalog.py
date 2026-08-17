"""Tests for remote catalog client and dual storage façade."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from kraken_hub.dual_catalog import DualCatalogService
from kraken_hub.remote_client import (
    RemoteAuth,
    RemoteHttpClient,
    RemoteServerError,
    RemoteServerProjectService,
)
from kraken_hub.workspace_service import REMOTE_STORAGE_PROFILE, project_from_server_dict
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.artifacts import ArtifactVersion, BlobRef
from kraken_manager.domain.common import ArtifactSeriesId, PrincipalId
from kraken_manager.infrastructure.filesystem._codec import encode_model
from kraken_manager.domain.project import (
    GridOrientation,
    LayerType,
    ProjectState,
    RepresentationKind,
)


class _FakeHttp(RemoteHttpClient):
    def __init__(self) -> None:
        super().__init__("http://example.test")
        self.calls: list[tuple[str, str, dict | None, dict]] = []
        self.responses: dict[tuple[str, str], object] = {}
        self.gateway_uploads: list[tuple[str, str, Path]] = []

    def request(self, method, path, *, token, payload=None, headers=None):
        self.calls.append((method.upper(), path, None if payload is None else dict(payload), dict(headers or {})))
        key = (method.upper(), path.split("?", 1)[0])
        if key not in self.responses:
            raise RemoteServerError(f"unexpected {method} {path}", status=500)
        value = self.responses[key]
        if isinstance(value, Exception):
            raise value
        return value

    def upload_url(self, url, *, token, source):
        self.gateway_uploads.append((url, token, source))
        return {"sha256": "uploaded"}


def test_project_from_server_dict_maps_fields() -> None:
    project = project_from_server_dict(
        {
            "project_id": "11111111-1111-1111-1111-111111111111",
            "name": "Shared",
            "width": 4,
            "height": 3,
            "orientation": "y_up",
            "state": "active",
            "revision": 2,
            "storage_profile": "server-postgres",
            "created_at": "2026-07-31T00:00:00+00:00",
        }
    )
    assert project.name == "Shared"
    assert project.width == 4
    assert project.orientation is GridOrientation.Y_UP
    assert project.state is ProjectState.ACTIVE
    assert project.revision == 2
    assert project.storage_profile == "server-postgres"


def test_managed_artifact_uses_gateway_transfer_contract(tmp_path: Path) -> None:
    http = _FakeHttp()
    project_id = str(uuid4())
    series_id = ArtifactSeriesId(str(uuid4()))
    source = tmp_path / "large.bin"
    source.write_bytes(b"gateway payload")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    auth = RemoteAuth(
        access_token="session",
        principal=Principal.local(subject="alice", display_name="Alice"),
    )
    created = ArtifactVersion.managed(
        series_id=series_id,
        blob=BlobRef(digest, source.stat().st_size),
        media_type="application/octet-stream",
        filename=source.name,
        author_principal_id=PrincipalId(str(auth.principal.id)),
        created_at=datetime.now(UTC),
    )
    http.responses[("GET", f"/api/v1/projects/{project_id}/artifacts/{series_id}/revision")] = {"revision": 0}
    http.responses[("POST", f"/api/v1/projects/{project_id}/artifacts/{series_id}/uploads")] = {
        "mode": "gateway",
        "url": f"https://blob.example.test/v1/blobs/{digest}",
        "token": "ticket",
    }
    http.responses[("POST", f"/api/v1/projects/{project_id}/artifacts/{series_id}/uploads/complete")] = (
        encode_model(created)
    )
    service = RemoteServerProjectService("https://server.example.test", auth=auth, data_dir=tmp_path, http=http)

    result = service.add_managed_artifact_version(
        principal=auth.principal,
        project_id=project_id,
        series_id=series_id,
        source=source,
        idempotency_key="upload-1",
    )

    assert result.sha256 == digest
    assert http.gateway_uploads == [(f"https://blob.example.test/v1/blobs/{digest}", "ticket", source)]
    complete = next(call for call in http.calls if call[1].endswith("/uploads/complete"))
    assert complete[2]["upload_token"] == "ticket"
    assert complete[3] == {"Idempotency-Key": "upload-1", "If-Match": "0"}


def test_remote_create_and_list_use_idempotency_header(tmp_path: Path) -> None:
    http = _FakeHttp()
    project_id = "22222222-2222-2222-2222-222222222222"
    http.responses[("POST", "/api/v1/projects")] = {
        "project_id": project_id,
        "name": "DB Project",
        "width": 2,
        "height": 2,
        "orientation": "y_down",
        "state": "active",
        "revision": 0,
        "storage_profile": "server-postgres",
        "created_at": datetime.now(UTC).isoformat(),
    }
    http.responses[("GET", "/api/v1/projects")] = {
        "items": [http.responses[("POST", "/api/v1/projects")]]
    }
    auth = RemoteAuth(
        access_token="token",
        principal=Principal.gitlab(
            issuer="https://gitlab.example",
            subject="42",
            display_name="Alice",
        ),
    )
    service = RemoteServerProjectService(
        "http://example.test",
        auth=auth,
        data_dir=tmp_path,
        http=http,
    )
    created = service.create_project(
        principal=auth.principal,
        name="DB Project",
        width=2,
        height=2,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="create-1",
        storage_profile_id=REMOTE_STORAGE_PROFILE.id,
    )
    assert created.id == project_id
    assert not service.supports_workspace_roots
    assert service.project_workspace(project_id) is None
    listed = service.list_projects()
    assert [item.name for item in listed] == ["DB Project"]
    method, path, payload, headers = http.calls[0]
    assert method == "POST"
    assert path == "/api/v1/projects"
    assert payload["name"] == "DB Project"
    assert headers["Idempotency-Key"] == "create-1"


def test_dual_catalog_routes_create_by_storage_profile(tmp_path: Path) -> None:
    from kraken_hub.composition import EmbeddedProjectService

    local = EmbeddedProjectService(data_dir=tmp_path / "local")
    account = local.create_initial_account("admin", "Admin", "secret")
    http = _FakeHttp()
    remote_id = str(uuid4())
    http.responses[("GET", "/api/v1/projects")] = {"items": []}
    http.responses[("POST", "/api/v1/projects")] = {
        "project_id": remote_id,
        "name": "Remote",
        "width": 1,
        "height": 1,
        "orientation": "y_down",
        "state": "active",
        "revision": 0,
        "storage_profile": REMOTE_STORAGE_PROFILE.id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    remote = RemoteServerProjectService(
        "http://example.test",
        auth=RemoteAuth(access_token="t", principal=account.principal),
        data_dir=tmp_path / "remote-cache",
        http=http,
    )
    # Remote auth should be gitlab for shared writes; dual still routes by profile id.
    remote.auth = RemoteAuth(
        access_token="t",
        principal=Principal.gitlab(
            issuer="https://gitlab.example",
            subject="1",
            display_name="Git",
        ),
    )
    dual = DualCatalogService(local, remote)
    profiles = {profile.id for profile in dual.list_storage_profiles()}
    assert local.profile.id in profiles
    assert REMOTE_STORAGE_PROFILE.id in profiles

    local_project = dual.create_project(
        principal=account.principal,
        name="LocalOne",
        width=1,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
        source_root=local.default_source_root,
        derived_root=local.default_derived_root,
        storage_profile_id=local.profile.id,
    )
    assert not dual.is_remote_project(local_project.id)
    assert dual.list_representations(local_project.id, uuid4()) == ()
    assert dual.project_storage_label(local_project.id) == "Локальный файл"

    remote_project = dual.create_project(
        principal=account.principal,
        name="Remote",
        width=1,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
        storage_profile_id=REMOTE_STORAGE_PROFILE.id,
    )
    http.responses[("GET", "/api/v1/projects")] = {
        "items": [http.responses[("POST", "/api/v1/projects")]]
    }
    assert dual.is_remote_project(remote_project.id)
    assert dual.project_storage_label(remote_project.id) == "Сервер PostgreSQL"
    names = {project.name for project in dual.list_projects()}
    assert names == {"LocalOne", "Remote"}


def test_dual_catalog_can_configure_remote_during_desktop_session(
    tmp_path: Path, monkeypatch
) -> None:
    from kraken_hub import dual_catalog
    from kraken_hub.composition import EmbeddedProjectService

    local = EmbeddedProjectService(data_dir=tmp_path / "local")
    account = local.create_initial_account("admin", "Admin", "secret")
    server_principal = Principal.gitlab(
        issuer="https://gitlab.example",
        subject="server-user",
        display_name="Server User",
    )
    instances = []

    class ConfiguredRemote:
        supports_live_sync = True

        def __init__(self, base_url, *, auth, data_dir):
            self.base_url = base_url
            self.auth = auth
            self.data_dir = data_dir
            self.started = False
            self.wake_handlers = []
            self.catalog_handlers = []
            self.status_handlers = []
            instances.append(self)

        def current_principal(self):
            return server_principal

        def list_projects(self, *, include_archived=False):
            del include_archived
            return ()

        def add_wake_handler(self, handler):
            self.wake_handlers.append(handler)

        def add_catalog_handler(self, handler):
            self.catalog_handlers.append(handler)

        def add_status_handler(self, handler):
            self.status_handlers.append(handler)

        def start_sync(self):
            self.started = True

        def stop_sync(self):
            self.started = False

    monkeypatch.setattr(dual_catalog, "RemoteServerProjectService", ConfiguredRemote)
    service = DualCatalogService(local)
    wake = lambda _event: None
    catalog = lambda: None
    status = lambda _value: None
    service.add_wake_handler(wake)
    service.add_catalog_handler(catalog)
    service.add_status_handler(status)
    service.start_sync()

    remote = service.configure_remote(
        base_url="https://kraken.example/",
        access_token="secret-token",
        principal=account.principal,
    )

    assert remote is instances[0]
    assert remote.base_url == "https://kraken.example"
    assert remote.auth.access_token == "secret-token"
    assert remote.auth.principal == server_principal
    assert remote.started
    assert remote.wake_handlers == [wake]
    assert remote.catalog_handlers == [catalog]
    assert remote.status_handlers == [status]


def test_remote_structure_routes_do_not_fall_back_to_local_catalog(
    tmp_path: Path,
) -> None:
    from kraken_hub.composition import EmbeddedProjectService

    local = EmbeddedProjectService(data_dir=tmp_path / "local")
    account = local.create_initial_account("admin", "Admin", "secret")
    remote_principal = Principal.gitlab(
        issuer="https://gitlab.example",
        subject="1",
        display_name="Git",
    )
    project_id = str(uuid4())
    layer_id = str(uuid4())
    representation_id = str(uuid4())
    project_payload = {
        "project_id": project_id,
        "name": "Remote",
        "width": 1,
        "height": 1,
        "orientation": "y_down",
        "state": "active",
        "revision": 0,
        "storage_profile": REMOTE_STORAGE_PROFILE.id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    layer_payload = {
        "layer_id": layer_id,
        "project_id": project_id,
        "name": "Metal",
        "type": "metal",
        "order": 0,
        "state": "active",
        "revision": 0,
    }
    representation_payload = {
        "representation_id": representation_id,
        "project_id": project_id,
        "layer_id": layer_id,
        "name": "Images",
        "kind": "image",
        "purpose": "source",
        "active": True,
        "state": "active",
        "revision": 0,
    }
    http = _FakeHttp()
    http.responses[("GET", "/api/v1/projects")] = {"items": [project_payload]}
    http.responses[("GET", "/api/v1/principals")] = {
        "items": [
            {
                "principal_id": str(remote_principal.id),
                "provider": "gitlab",
                "issuer": remote_principal.issuer,
                "subject": remote_principal.subject,
                "display_name": remote_principal.display_name,
                "email": None,
                "active": True,
                "system_roles": [],
            }
        ]
    }
    http.responses[("POST", f"/api/v1/projects/{project_id}/layers")] = layer_payload
    http.responses[("GET", f"/api/v1/projects/{project_id}/layers")] = {
        "items": [layer_payload]
    }
    http.responses[
        ("POST", f"/api/v1/projects/{project_id}/layers/{layer_id}/representations")
    ] = representation_payload
    http.responses[
        ("GET", f"/api/v1/projects/{project_id}/layers/{layer_id}/representations")
    ] = {"items": [representation_payload]}
    http.responses[
        (
            "PATCH",
            f"/api/v1/projects/{project_id}/layers/{layer_id}/representations/{representation_id}",
        )
    ] = {**representation_payload, "active": False, "revision": 1}
    remote = RemoteServerProjectService(
        "http://example.test",
        auth=RemoteAuth(access_token="t", principal=remote_principal),
        data_dir=tmp_path / "remote-cache",
        http=http,
    )
    dual = DualCatalogService(local, remote)
    project = project_from_server_dict(project_payload)

    layer = dual.create_layer(
        principal=account.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=0,
        idempotency_key="layer",
    )
    representation = dual.create_representation(
        principal=account.principal,
        project=project,
        layer=layer,
        name="Images",
        kind=RepresentationKind.IMAGE,
        active=True,
        idempotency_key="representation",
    )

    assert layer.id == layer_id
    assert representation.id == representation_id
    listed_representations = dual.list_representations(project.id, layer.id)
    assert tuple(item.id for item in listed_representations) == (representation.id,)
    assert listed_representations[0].active
    deactivated = dual.deactivate_representation(
        principal=account.principal,
        project=project,
        layer=layer,
        representation=representation,
        idempotency_key="deactivate",
    )
    assert not deactivated.active
    assert dual.list_project_principals(project.id) == (remote_principal,)
    patch_call = next(call for call in http.calls if call[0] == "PATCH")
    assert patch_call[2] == {
        "active": False,
        "expected_representation_revision": 0,
    }
    assert local.list_layers(project.id) == ()
    layer_call = next(
        call for call in http.calls if call[1].endswith(f"/{project_id}/layers") and call[0] == "POST"
    )
    assert layer_call[3] == {"Idempotency-Key": "layer", "If-Match": "0"}
