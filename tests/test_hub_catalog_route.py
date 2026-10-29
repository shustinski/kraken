"""Dual catalog sends a project call to the backend that owns it."""

from __future__ import annotations

from pathlib import Path

import pytest

from kraken_hub.dual_catalog import DualCatalogService
from kraken_hub.remote_client import RemoteServerError


class _Project:
    def __init__(self, project_id: str) -> None:
        self.id = project_id


class _Local:
    def __init__(self) -> None:
        self.data_dir = Path(".")
        self.profile = object()
        self.calls: list[str] = []

    def plan_import_directory(self, **_kwargs):
        self.calls.append("plan_import_directory")
        return "local-plan"

    def open_owned(self, **_kwargs):
        self.calls.append("open_owned")
        return "local-owned"

    def missing_operation(self, **_kwargs):
        self.calls.append("missing_operation")
        return "local-missing"


class _Remote:
    def __init__(self, projects: tuple[_Project, ...] = ()) -> None:
        self._projects = projects
        self.calls: list[str] = []

    def list_projects(self, *, include_archived: bool = False):
        return self._projects

    def plan_import_directory(self, **_kwargs):
        self.calls.append("plan_import_directory")
        return "remote-plan"

    def open_owned(self, **_kwargs):
        self.calls.append("open_owned")
        return "remote-owned"


def test_known_server_project_is_labeled_and_routed_to_the_server() -> None:
    local = _Local()
    remote = _Remote((_Project("remote-1"),))
    catalog = DualCatalogService(local, remote)

    assert catalog.is_remote_project("remote-1") is True
    assert catalog.is_remote_project("local-1") is False
    assert catalog.project_storage_label("remote-1") == "Сервер PostgreSQL"
    assert catalog.project_storage_label("local-1") == "Локальный файл"
    assert catalog.open_owned(project_id="remote-1") == "remote-owned"
    assert remote.calls == ["open_owned"]
    assert local.calls == []


def test_directory_plan_stays_local_even_for_a_server_project() -> None:
    local = _Local()
    remote = _Remote((_Project("remote-1"),))
    catalog = DualCatalogService(local, remote)

    assert catalog.plan_import_directory(project_id="remote-1") == "local-plan"
    assert local.calls == ["plan_import_directory"]
    assert remote.calls == []


def test_server_project_without_that_operation_is_refused() -> None:
    local = _Local()
    remote = _Remote((_Project("remote-1"),))
    catalog = DualCatalogService(local, remote)

    with pytest.raises(RemoteServerError, match="не подключена"):
        catalog.missing_operation(project=_Project("remote-1"))
    assert local.calls == []


def test_unknown_project_uses_the_local_catalog() -> None:
    local = _Local()
    catalog = DualCatalogService(local, _Remote(()))

    assert catalog.open_owned(project_id="local-1") == "local-owned"
    assert local.calls == ["open_owned"]
