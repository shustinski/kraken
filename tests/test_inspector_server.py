"""Run the production server composition on a disposable PostgreSQL cluster."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from kraken_inspector.session import InspectorSession
from kraken_hub.infrastructure.remote.client import RemoteServerProjectService


def test_inspector_owns_real_postgres_server_and_connected_hub() -> None:
    postgres_bin = Path(os.environ.get("KRAKEN_TEST_PG_BIN", r"C:\Program Files\PostgreSQL\18\bin"))
    if not (postgres_bin / "initdb.exe").is_file():
        pytest.skip("PostgreSQL binaries are not installed")
    workspace = Path(__file__).resolve().parents[1]
    session = InspectorSession(workspace)
    previous = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    try:
        agent = session.start_agent()
        assert agent["health"]["status"] == "ok"
        agent_run_dir = session.run_dir
        server = session.start_server(postgres_bin=postgres_bin)
        assert session.run_dir == agent_run_dir
        assert server["health"]["status"] == "ok"
        assert server["health"]["metadata"] == "postgresql"
        hub = session.launch_hub(timeout=40)
        assert hub["server_mode"]
        ui = session.inspect_ui()
        assert any("сервер" in node["text"].lower() for window in ui["windows"] for node in window["nodes"])
        assert any(
            item["name"] == "server_healthy" and item["status"] == "pass"
            for item in session.evaluate_invariants()["checks"]
        )
        assert any(
            item["name"] == "agent_healthy" and item["status"] == "pass"
            for item in session.evaluate_invariants()["checks"]
        )
        create_button = next(
            node
            for window in ui["windows"]
            for node in window["nodes"]
            if node["class"] == "QPushButton" and node["text"] == "Создать проект"
        )
        assert session.perform_action(create_button["id"], "click")["ok"]
        assert session.wait_for_ui(window_title="Новый проект", timeout=5)["found"]
        dialog = next(window for window in session.inspect_ui()["windows"] if window["title"] == "Новый проект")
        name_field = next(node for node in dialog["nodes"] if node["object_name"] == "projectName")
        assert session.perform_action(name_field["id"], "type", "Inspector remote project")["ok"]
        assert session.wait_for_ui(text="Inspector remote project", timeout=5)["found"]
        dialog = next(window for window in session.inspect_ui()["windows"] if window["title"] == "Новый проект")
        confirm = next(
            node for node in dialog["nodes"] if node["class"] == "QPushButton" and node["text"] == "Создать проект"
        )
        assert session.perform_action(confirm["id"], "click")["ok"]
        managed = session._managed_server
        assert managed is not None
        remote = RemoteServerProjectService.authenticate_local(
            managed.url,
            username=managed.username,
            password=managed.password,
            data_dir=session.run_dir / "independent-client",
        )
        try:
            for _ in range(50):
                if any(project.name == "Inspector remote project" for project in remote.list_projects()):
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("GUI-created project is absent from the real server")
        finally:
            remote.stop_sync()
    finally:
        session.stop()
        session.stop_agent()
        session.stop_server()
        if previous is None:
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = previous
