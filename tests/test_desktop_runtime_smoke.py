"""Runtime smoke for Desktop against optional live Server/Agent."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

from kraken_agent.runner import PluginRegistry
from kraken_hub import manager_app
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.dual_catalog import DualCatalogService
from kraken_hub.manager_app import (
    DesktopController,
    _configure_administration_page,
    _my_work_panel,
    _performer_panel,
    _plugin_panel,
    _statistics_panel,
)
from kraken_hub.remote_client import RemoteHttpClient
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation
from kraken_manager.presentation.qt import ProjectListItem, ProjectManagerShell


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_plugin_registry_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "plugins.json"
    path.write_bytes(b'\xef\xbb\xbf{"plugins":[]}')
    assert PluginRegistry.from_json(path).operations() == frozenset()


def test_desktop_controller_opens_local_project(qapp, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok),
    )

    class FakeRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            self.process = None
            self.base_url = ""
            self.token = ""

        def ensure_started(self):
            return frozenset()

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(manager_app, "LocalAgentRuntime", FakeRuntime)

    service = EmbeddedProjectService(tmp_path)
    session = service.create_initial_account("admin", "Administrator", "")
    sources = tmp_path / "sources"
    derived = tmp_path / "derived"
    sources.mkdir()
    derived.mkdir()
    project = service.create_project(
        principal=session.principal,
        name="Smoke",
        width=4,
        height=3,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
        layer_template=True,
        source_root=sources,
        derived_root=derived,
    )
    shell = ProjectManagerShell()
    controller = DesktopController(
        shell, service, session, thumbnail_store_uri="memory://", plugin_items=[]
    )
    controller._agent_timer.stop()
    controller.open_project(
        ProjectListItem(str(project.id), project.name, project.width, project.height, "local")
    )
    assert controller._workspace is not None
    qapp.processEvents()

    _configure_administration_page(shell, service, session)
    page = shell.page("administration")
    assert "Проверить целостность" in [button.text() for button in page.findChildren(QPushButton)]
    assert _performer_panel(service) is not None
    assert _my_work_panel(service, session, controller) is not None
    assert _statistics_panel(service) is not None
    assert _plugin_panel([]) is not None

    DesktopController._map_not_configured(controller._workspace)
    controller._agent_timer.stop()
    controller._workspace = None


def test_live_server_dual_catalog_smoke(tmp_path: Path) -> None:
    try:
        health = RemoteHttpClient("http://127.0.0.1:8080").request(
            "GET", "/api/v1/health", token=""
        )
    except Exception as exc:  # noqa: BLE001 - optional live dependency
        pytest.skip(f"kraken-server not available: {exc}")
    if health.get("status") != "ok":
        pytest.skip("kraken-server unhealthy")

    local = EmbeddedProjectService(tmp_path)
    local.create_initial_account("admin", "Administrator", "")
    dual = DualCatalogService(local)
    dual.configure_remote(
        base_url="http://127.0.0.1:8080",
        access_token="smoke-dev-token",
        principal=Principal.local(subject="smoke", display_name="Smoke"),
    )
    created = dual.create_project(
        principal=dual.remote.auth.principal,
        name=f"Smoke Shared {uuid4().hex[:8]}",
        width=6,
        height=4,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key=str(uuid4()),
        storage_profile_id="server-postgres",
    )
    assert created is not None
    assert any(project.id == created.id for project in dual.list_projects())


def test_live_agent_health_smoke() -> None:
    from urllib.request import Request, urlopen

    try:
        request = Request(
            "http://127.0.0.1:8765/api/v1/health",
            headers={"Authorization": "Bearer qa-agent-token"},
        )
        with urlopen(request, timeout=2) as response:
            body = response.read().decode()
    except Exception as exc:  # noqa: BLE001 - optional live dependency
        pytest.skip(f"kraken-agent not available: {exc}")
    assert '"status"' in body and "ok" in body
