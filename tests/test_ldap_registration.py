"""Registration pre-check: a login held by a domain principal is rejected.

Task 8: the HTTP registration route (409) and the shared admin application
function (ValueError) both refuse to create a local account whose login is
claimed by a domain (LDAP/AD) principal. Password reset refuses domain
accounts, and the account rows expose the auth provider. The pre-check is a
pure DB lookup — AD is never consulted here.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_server.app import create_app
from kraken_server.composition import HybridSessionResolver

pytest_plugins = ["tests.test_admin_project_workflows"]


def _hold_login(services, login: str, display: str = "T") -> None:
    services.identities.save(
        Principal.ldap(
            issuer="ldap://fake:389",
            subject=login,
            display_name=display,
            principal_id=str(uuid4()),
            login=login,
        )
    )


def _client(services, accounts):
    # identities MUST be passed, otherwise the pre-check is disabled
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    return TestClient(create_app(services=services, account_store=accounts,
                                 session_resolver=resolver, project_access_mode="acl",
                                 identities=services.identities))


def test_http_registration_rejects_ldap_held_login(database):
    services, accounts, _o, _c = database
    _hold_login(services, "taken")
    resp = _client(services, accounts).post(
        "/api/v1/auth/accounts", json={"username": "taken", "password": "pw"}
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "Логин «taken» уже занят"


def test_http_registration_still_allows_new_login(database):
    services, accounts, _o, _c = database
    _hold_login(services, "taken")
    resp = _client(services, accounts).post(
        "/api/v1/auth/accounts", json={"username": "free.user", "password": "pw"}
    )
    assert resp.status_code == 201, resp.text


def test_cli_create_account_rejects_ldap_held_login(database):
    from kraken_admin.application.accounts import create_account

    services, accounts, _o, _c = database
    _hold_login(services, "taken2", display="T2")
    with pytest.raises(ValueError, match="уже занят"):
        create_account(accounts, "taken2", "T2", "pw", identities=services.identities)


def test_cli_create_account_without_identities_still_works(database):
    from kraken_admin.application.accounts import create_account

    services, accounts, _o, _c = database
    account_id = create_account(accounts, "plain.local", "PL", "pw")
    assert account_id


def test_reset_password_refuses_ldap_account(database):
    from kraken_admin.application.accounts import reset_password

    services, accounts, _o, _c = database
    accounts.create_ldap_account("rp.user", "RP")
    with pytest.raises(ValueError, match="доменом"):
        reset_password(accounts, "rp.user", "newpw")


def test_account_rows_include_provider(database):
    from kraken_admin.application.accounts import account_rows

    services, accounts, _o, _c = database
    accounts.create_ldap_account("prov.user", "P")
    rows = {r["username"]: r["auth_provider"] for r in account_rows(accounts)}
    assert rows["prov.user"] == "ldap"
    assert rows["admin"] == "local"
