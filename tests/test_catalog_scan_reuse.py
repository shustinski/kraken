"""One catalog refresh reuses the project list and each local projection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.composition.parts.create_artifact_series import SQLiteProjectionStore
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService
from kraken_hub.application.workspace import REMOTE_STORAGE_PROFILE
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation

from test_remote_catalog import _FakeHttp


def _remote_project(project_id: str) -> dict[str, object]:
    return {
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


def _remote_service(tmp_path: Path, http: _FakeHttp) -> RemoteServerProjectService:
    return RemoteServerProjectService(
        "http://example.test",
        auth=RemoteAuth(
            access_token="token",
            principal=Principal.gitlab(
                issuer="https://gitlab.example",
                subject="1",
                display_name="Git",
            ),
        ),
        data_dir=tmp_path / "remote-cache",
        http=http,
    )


def test_dual_catalog_refreshes_remote_ids_with_one_project_list(tmp_path: Path) -> None:
    project_id = str(uuid4())
    http = _FakeHttp()
    http.responses[("GET", "/api/v1/projects")] = {"items": [_remote_project(project_id)]}
    local = EmbeddedProjectService(tmp_path / "local")
    local.create_initial_account("owner", "Owner", "secret")
    dual = DualCatalogService(local, _remote_service(tmp_path, http))

    http.calls.clear()
    listed = dual.list_projects(include_archived=True)

    assert [project.name for project in listed] == ["Remote"]
    assert [path for method, path, *_rest in http.calls] == ["/api/v1/projects?include_archived=true"]
    assert dual.is_remote_project(project_id)


def test_remote_work_reads_use_the_supplied_project_list(tmp_path: Path) -> None:
    project_id = str(uuid4())
    http = _FakeHttp()
    http.responses[("GET", "/api/v1/projects")] = {"items": [_remote_project(project_id)]}
    http.responses[("GET", f"/api/v1/projects/{project_id}/jobs")] = {"items": []}
    http.responses[("GET", f"/api/v1/projects/{project_id}/work-orders")] = {"items": []}
    http.responses[("GET", f"/api/v1/projects/{project_id}/reviews")] = {"items": []}
    service = _remote_service(tmp_path, http)
    projects = service.list_projects(include_archived=True)

    http.calls.clear()
    assert service.plugin_jobs(projects=projects) == ()
    assert service.work_orders(projects=projects) == ()
    assert service.review_batches(projects=projects) == ()

    assert [path for method, path, *_rest in http.calls] == [
        f"/api/v1/projects/{project_id}/jobs",
        f"/api/v1/projects/{project_id}/work-orders",
        f"/api/v1/projects/{project_id}/reviews",
    ]


def test_local_catalog_scan_reopens_a_projection_only_when_the_log_moves(tmp_path: Path, monkeypatch) -> None:
    service = EmbeddedProjectService(tmp_path / "local")
    session = service.create_initial_account("owner", "Owner", "secret")
    service.create_project(
        principal=session.principal,
        name="Chip",
        width=1,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project",
    )
    opened: list[str] = []
    real_init = SQLiteProjectionStore.__init__

    def counted(store: SQLiteProjectionStore, catalog_root: object, project_id: str) -> None:
        opened.append(str(project_id))
        real_init(store, catalog_root, project_id)

    monkeypatch.setattr(SQLiteProjectionStore, "__init__", counted)
    projects = service.list_projects(include_archived=True)
    opened.clear()

    service.plugin_jobs(projects=projects)
    service.work_orders(projects=projects)
    service.review_batches(projects=projects)

    assert opened == []
