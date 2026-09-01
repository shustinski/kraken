from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.cli import _database_url
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


def test_server_admin_manages_accounts_roles_sessions_and_audit(tmp_path: Path) -> None:
    client, store, administrator, operator = _administration_client(tmp_path)
    admin_headers = {"Authorization": "Bearer admin-token"}

    denied = client.get("/api/v1/admin/accounts", headers={"Authorization": "Bearer operator-token"})
    assert denied.status_code == 403

    created = client.post(
        "/api/v1/admin/accounts",
        headers=admin_headers,
        json={"username": "reviewer", "display_name": "Reviewer", "password": "review password"},
    )
    assert created.status_code == 201
    reviewer_id = created.json()["account_id"]

    granted = client.put(
        f"/api/v1/admin/accounts/{operator.account_id}/roles/server_admin",
        headers=admin_headers,
    )
    assert granted.status_code == 200
    assert granted.json()["system_roles"] == ["server_admin"]

    disabled = client.post(f"/api/v1/admin/accounts/{reviewer_id}/disable", headers=admin_headers, json={})
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False

    reset = client.put(
        f"/api/v1/admin/accounts/{operator.account_id}/password",
        headers=admin_headers,
        json={"password": "replacement password"},
    )
    assert reset.status_code == 200
    assert store.authenticate("operator", "replacement password") is not None

    audit = client.get("/api/v1/admin/audit", headers=admin_headers)
    assert audit.status_code == 200
    actions = {item["action"] for item in audit.json()["items"]}
    assert {
        "account.created",
        "account.disabled",
        "account.password_reset",
        "global_role.granted",
    } <= actions

    session = client.get("/api/v1/session", headers=admin_headers)
    assert session.json()["system_roles"] == ["server_admin"]
    assert administrator.account_id != operator.account_id


def test_administrator_cannot_disable_or_demote_self(tmp_path: Path) -> None:
    client, _store, administrator, _operator = _administration_client(tmp_path)
    headers = {"Authorization": "Bearer admin-token"}
    assert (
        client.post(
            f"/api/v1/admin/accounts/{administrator.account_id}/disable",
            headers=headers,
            json={},
        ).status_code
        == 409
    )
    assert (
        client.delete(
            f"/api/v1/admin/accounts/{administrator.account_id}/roles/server_admin",
            headers=headers,
        ).status_code
        == 409
    )


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
        host="127.0.0.1",
        port=9080,
        project_access_mode="acl",
    )
    loaded = ServerConfig.load(config.path)
    assert loaded.database_url == database_url
    assert loaded.port == 9080
    assert loaded.blob_root == (tmp_path / "blobs").resolve()
    generated = config.path.read_text(encoding="utf-8")
    assert "никогда не записывайте пароль" in generated
    assert "Windows DPAPI" in generated
    assert "Неизменяемые файлы сервера" in generated


def test_initial_setup_accepts_a_new_config_path(monkeypatch, tmp_path: Path) -> None:
    prompted_url = "postgresql+psycopg://kraken@localhost/kraken"
    monkeypatch.setattr("kraken_server.cli.getpass.getpass", lambda _prompt: prompted_url)

    assert _database_url(Namespace(config=tmp_path / "new-server.toml", database_url=None)) == prompted_url
