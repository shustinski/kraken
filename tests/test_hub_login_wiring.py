"""Regression: run_manager_gui must hand the full workspace service to _login.

The login window's domain/server tabs call ``service.configure_remote_local``,
which exists only on the composition wrapper ``DualCatalogService``. Passing
``service.local_administration`` (the bare ``EmbeddedProjectService``) broke
server sign-in with ``AttributeError`` while the local tab kept working, so
the dialog-only tests could not see it.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

import kraken_hub.presentation.qt.parts.run_manager_gui as gui_parts
from kraken_hub.composition.catalog import build_workspace_service

_ISOLATED_ENV = ("KRAKEN_SERVER_URL", "KRAKEN_GITLAB_TOKEN", "KRAKEN_SERVER_TOKEN", "KRAKEN_UPDATE_URL")


def test_run_manager_gui_passes_full_service_to_login(tmp_path, monkeypatch):
    """_login receives the DualCatalogService; _development_session keeps the local backend."""
    for name in _ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
    service = build_workspace_service(data_dir=tmp_path)
    assert hasattr(service, "configure_remote_local")
    assert not hasattr(service.local_administration, "configure_remote_local")

    seen: dict = {}

    def fake_development_session(local):
        seen["development_local"] = local
        return None

    def fake_login(parent, received_service):
        seen["login_service"] = received_service
        return None  # no session: run_manager_gui returns before building the shell

    monkeypatch.setattr(gui_parts, "_development_session", fake_development_session)
    monkeypatch.setattr(gui_parts, "_login", fake_login)

    exit_code = gui_parts.run_manager_gui(
        [], service=service, application_version="0", plugin_launcher=None
    )

    assert exit_code == 1
    assert seen["login_service"] is service, (
        "_login needs the full workspace service: its server/domain tabs call configure_remote_local"
    )
    assert seen["development_local"] is service.local_administration
