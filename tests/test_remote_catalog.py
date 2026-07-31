"""Tests for remote catalog client and dual storage façade."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from kraken_hub.dual_catalog import DualCatalogService
from kraken_hub.remote_client import (
    RemoteAuth,
    RemoteHttpClient,
    RemoteServerError,
    RemoteServerProjectService,
)
from kraken_hub.workspace_service import REMOTE_STORAGE_PROFILE, project_from_server_dict
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation, ProjectState


class _FakeHttp(RemoteHttpClient):
    def __init__(self) -> None:
        super().__init__("http://example.test")
        self.calls: list[tuple[str, str, dict | None, dict]] = []
        self.responses: dict[tuple[str, str], object] = {}

    def request(self, method, path, *, token, payload=None, headers=None):
        self.calls.append((method.upper(), path, None if payload is None else dict(payload), dict(headers or {})))
        key = (method.upper(), path.split("?", 1)[0])
        if key not in self.responses:
            raise RemoteServerError(f"unexpected {method} {path}", status=500)
        value = self.responses[key]
        if isinstance(value, Exception):
            raise value
        return value


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
