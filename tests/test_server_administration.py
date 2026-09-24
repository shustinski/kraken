from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
from kraken_server.app import SessionPrincipal, create_app
from kraken_admin.cli import _database_url
from kraken_server.configuration import ServerConfig, protect_secret, unprotect_secret, write_config
from kraken_server.services import InMemoryServerServices


def _administration_client(tmp_path: Path):
    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    administrator = store.create_account("admin", "Administrator", "admin password")
    store.grant_global_role(administrator.account_id, "server_admin")
    operator = store.create_account("operator", "Operator", "operator password")

    def resolve(token: str):
        mapping = {"admin-token": administrator.account_id, "operator-token": operator.account_id}
        account_id = mapping.get(token)
        return None if account_id is None else SessionPrincipal(account_id, "local", token)

    app = create_app(
        services=InMemoryServerServices(),
        account_store=store,
        session_resolver=resolve,
        project_access_mode="acl",
    )
    return TestClient(app), store, administrator, operator


def test_local_admin_manages_accounts_without_http(tmp_path: Path) -> None:
    client, store, administrator, operator = _administration_client(tmp_path)
    from kraken_admin.local_admin import (
        create_account,
        reset_password,
        set_account_enabled,
        set_server_admin,
    )

    assert client.get("/api/v1/admin/accounts", headers={"Authorization": "Bearer admin-token"}).status_code == 404

    reviewer_id = create_account(store, "reviewer", "Reviewer", "review password")
    set_server_admin(store, "operator", grant=True)
    assert store.global_roles_for(operator.account_id) == frozenset({"server_admin"})
    set_account_enabled(store, "reviewer", enabled=False)
    assert store.get_account(reviewer_id).enabled is False
    reset_password(store, "operator", "replacement password")
    assert store.authenticate("operator", "replacement password") is not None
    assert administrator.account_id != operator.account_id


def test_local_admin_cannot_remove_the_last_administrator(tmp_path: Path) -> None:
    _client, store, _administrator, _operator = _administration_client(tmp_path)
    from kraken_admin.local_admin import set_account_enabled, set_server_admin

    with pytest.raises(ValueError, match="last active"):
        set_account_enabled(store, "admin", enabled=False)
    with pytest.raises(ValueError, match="last active"):
        set_server_admin(store, "admin", grant=False)


def test_account_store_preserves_last_enabled_administrator(tmp_path: Path) -> None:
    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    administrator = store.create_account("admin", "Administrator", "admin password")
    store.grant_global_role(administrator.account_id, "server_admin")

    with pytest.raises(ValueError, match="last active"):
        store.revoke_global_role(
            administrator.account_id,
            "server_admin",
            preserve_last_enabled=True,
        )
    with pytest.raises(ValueError, match="last active"):
        store.set_enabled(
            administrator.account_id,
            False,
            preserve_last_admin=True,
        )

    assert store.global_roles_for(administrator.account_id) == frozenset({"server_admin"})
    assert store.get_account(administrator.account_id).enabled is True


def test_packaged_configuration_round_trip(tmp_path: Path) -> None:
    secret = tmp_path / "database.secret"
    database_url = "postgresql+psycopg://kraken:secret@db/kraken"
    protect_secret(database_url, secret)
    assert unprotect_secret(secret) == database_url

    config = write_config(
        tmp_path / "server.toml",
        database_url=database_url,
        blob_root=tmp_path / "blobs",
        source_root=tmp_path / "source",
        derived_root=tmp_path / "derived",
        host="127.0.0.1",
        port=9080,
        project_access_mode="acl",
    )
    loaded = ServerConfig.load(config.path)
    assert loaded.database_url == database_url
    assert loaded.port == 9080
    assert loaded.blob_root == (tmp_path / "blobs").resolve()
    assert loaded.source_root == (tmp_path / "source").resolve()
    assert loaded.derived_root == (tmp_path / "derived").resolve()
    generated = config.path.read_text(encoding="utf-8")
    assert "никогда не записывайте пароль" in generated
    assert "Windows DPAPI" in generated
    assert "Неизменяемые файлы сервера" in generated


def test_server_data_roots_may_be_the_same_directory(tmp_path: Path) -> None:
    shared = tmp_path / "data"
    config = write_config(
        tmp_path / "server.toml",
        database_url="postgresql+psycopg://kraken:secret@db/kraken",
        blob_root=tmp_path / "blobs",
        source_root=shared,
        derived_root=shared,
    )
    loaded = ServerConfig.load(config.path)
    assert loaded.source_root == loaded.derived_root == shared.resolve()


def test_self_registered_account_is_ordinary_and_sees_only_its_projects(tmp_path: Path) -> None:
    store = LocalAccountStore(tmp_path / "accounts.sqlite3", ScryptPasswordHasher())
    app = create_app(
        services=InMemoryServerServices(),
        account_store=store,
        project_access_mode="acl",
    )
    client = TestClient(app)

    def register(username: str, password: str) -> dict:
        response = client.post(
            "/api/v1/auth/accounts",
            json={"username": username, "password": password, "display_name": username},
        )
        assert response.status_code == 201, response.text
        return response.json()

    alice = register("alice", "secret")
    assert store.global_roles_for(alice["principal"]["id"]) == frozenset()
    created = client.post(
        "/api/v1/projects",
        headers={
            "Authorization": f"Bearer {alice['access_token']}",
            "Idempotency-Key": "alice-project",
        },
        json={"name": "Проект Alice", "width": 2, "height": 2},
    )
    assert created.status_code == 201, created.text

    bob = register("bob", "secret")
    visible = client.get(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {bob['access_token']}"},
    )
    assert visible.status_code == 200
    assert visible.json()["items"] == []
    denied = client.get(
        "/api/v1/admin/accounts",
        headers={"Authorization": f"Bearer {bob['access_token']}"},
    )
    assert denied.status_code == 404

    alice_projects = client.get(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {alice['access_token']}"},
    )
    assert [item["name"] for item in alice_projects.json()["items"]] == ["Проект Alice"]
    assert client.post(
        "/api/v1/auth/accounts",
        json={"username": "alice", "password": "wrong", "display_name": "Alice"},
    ).status_code == 401
    again = register("alice", "secret")
    assert again["principal"]["id"] == alice["principal"]["id"]


def test_local_admin_revokes_the_last_maintainer(tmp_path: Path) -> None:
    from dataclasses import replace
    from uuid import uuid4

    from kraken_hub.composition import EmbeddedProjectService
    from kraken_manager.application.acl import RevokeProjectRoleHandler
    from kraken_manager.application.dto import CommandContext, RevokeProjectRoleCommand
    from kraken_manager.domain.identity import ProjectRole, SystemRole
    from kraken_manager.domain.project import GridOrientation

    service = EmbeddedProjectService(tmp_path)
    owner = service.create_initial_account("owner", "Owner", "")
    project = service.create_project(
        principal=owner.principal,
        name="Protected owner",
        width=1,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project",
    )
    actor = replace(owner.principal, system_roles=frozenset({SystemRole.SERVER_ADMIN}))
    RevokeProjectRoleHandler(service._uow(str(project.id)), service.profiles, service.clock)(
        RevokeProjectRoleCommand(
            context=CommandContext(actor=actor, idempotency_key=str(uuid4())),
            project_id=project.id,
            principal_id=owner.principal.id,
            role=ProjectRole.MAINTAINER,
            expected_revision=service.project_role_revision(project.id, owner.principal.id),
        )
    )
    assert ProjectRole.MAINTAINER not in service.project_roles(project.id, owner.principal.id)


def test_initial_setup_accepts_a_new_config_path(monkeypatch, tmp_path: Path) -> None:
    prompted_url = "postgresql+psycopg://kraken@localhost/kraken"
    monkeypatch.setattr("kraken_admin.cli.getpass.getpass", lambda _prompt: prompted_url)

    assert _database_url(Namespace(config=tmp_path / "new-server.toml", database_url=None)) == prompted_url
