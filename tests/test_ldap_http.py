"""HTTP wiring for domain (LDAP/AD) sign-in.

Task 7: the login route uses the DomainAuthBroker when one is injected,
local registration is gated for in-LAN clients, and the "ldap" provider is
accepted by the mutation gate, the system-roles lookup and the session
fallback. AD is consulted only at login, never on an authenticated path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_manager.infrastructure.auth.ldap import LdapUnavailable
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_server.app import create_app
from kraken_server.composition import HybridSessionResolver
from tests.ldap_fixtures import FakeLdap, _ldap_identity, make_broker

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_is_lan_client_matrix():
    from kraken_server.interface.http.ip_gate import is_lan_client

    assert is_lan_client("10.1.2.3", ["10.0.0.0/8"]) is True
    assert is_lan_client("8.8.8.8", ["10.0.0.0/8", "192.168.0.0/16"]) is False
    assert is_lan_client("192.168.0.1", ["192.168.0.0/16"]) is True
    assert is_lan_client("192.168.1.255", ["192.168.0.0/16"]) is True
    assert is_lan_client("10.0.0.0", ["10.0.0.0/8"]) is True  # network address, strict=False
    assert is_lan_client("2001:db8::1", ["2001:db8::/32"]) is True
    assert is_lan_client("10.1.2.3", []) is False
    assert is_lan_client("not-an-ip", ["10.0.0.0/8"]) is False


def test_domain_login_end_to_end(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({"e2e": _ldap_identity("e2e", display="E2E User", email="e2e@inme")})
    broker = make_broker(accounts, services.identities, fake, PostgresPerformerStore(services.engine))
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=resolver, project_access_mode="acl",
                                   domain_auth=broker))
    resp = client.post("/api/v1/auth/sessions", json={"username": "e2e", "password": "pw"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    me = client.get("/api/v1/session", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert me.json()["display_name"] == "E2E User"


def test_ad_unavailable_returns_503(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap(verify_error=LdapUnavailable("dc down"))
    broker = make_broker(accounts, services.identities, fake, None)
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=resolver, project_access_mode="acl",
                                   domain_auth=broker))
    resp = client.post("/api/v1/auth/sessions", json={"username": "ghost", "password": "pw"})
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == "Доменная авторизация временно недоступна"


def test_local_login_unchanged_when_broker_present(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({})
    broker = make_broker(accounts, services.identities, fake, None)
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=resolver, project_access_mode="acl",
                                   domain_auth=broker))
    resp = client.post("/api/v1/auth/sessions", json={"username": "admin", "password": "password"})
    assert resp.status_code == 200, resp.text
    assert fake.calls == []


def test_registration_blocked_from_lan(database):
    services, accounts, _owner, _config = database
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    blocked = TestClient(create_app(services=services, account_store=accounts,
                                    session_resolver=resolver, project_access_mode="acl",
                                    lan_gate=lambda ip: True))
    resp = blocked.post("/api/v1/auth/accounts", json={"username": "lan-user", "password": "secret"})
    assert resp.status_code == 403, resp.text
    detail = (
        "Создание локальных аккаунтов из внутренней сети запрещено. "
        "Используйте доменную учётную запись."
    )
    assert resp.json()["detail"] == detail

    allowed = TestClient(create_app(services=services, account_store=accounts,
                                    session_resolver=resolver, project_access_mode="acl",
                                    lan_gate=lambda ip: False))
    resp = allowed.post("/api/v1/auth/accounts", json={"username": "lan-user", "password": "secret"})
    assert resp.status_code == 201, resp.text


def test_ldap_token_passes_mutation_gate(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({"gate.user": _ldap_identity("gate.user", display="Gate User")})
    broker = make_broker(accounts, services.identities, fake, PostgresPerformerStore(services.engine))
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=resolver, project_access_mode="acl",
                                   domain_auth=broker))
    login = client.post("/api/v1/auth/sessions", json={"username": "gate.user", "password": "pw"})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    account = accounts.get_by_username("gate.user")
    accounts.grant_global_role(account.account_id, "server_admin")
    created = client.post(
        "/api/v1/projects",
        json={"name": "Ldap Project", "width": 2, "height": 2},
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "ldap-create-project"},
    )
    assert created.status_code == 201, created.text


def test_ldap_session_reports_granted_global_roles(database):
    """An LDAP principal's system roles come from the identity catalog (uuid5 id)."""
    services, accounts, _owner, _config = database
    fake = FakeLdap({"role.user": _ldap_identity("role.user", display="Role User")})
    broker = make_broker(accounts, services.identities, fake, PostgresPerformerStore(services.engine))
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=resolver, project_access_mode="acl",
                                   domain_auth=broker))
    login = client.post("/api/v1/auth/sessions", json={"username": "role.user", "password": "pw"})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    account = accounts.get_by_username("role.user")
    accounts.grant_global_role(account.account_id, "server_admin")
    me = client.get("/api/v1/session", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert "server_admin" in me.json()["system_roles"]


def test_existing_session_survives_ad_outage(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({"outage.user": _ldap_identity("outage.user", display="Outage User")})
    broker = make_broker(accounts, services.identities, fake, PostgresPerformerStore(services.engine))
    resolver = HybridSessionResolver(accounts, services.identities, None,
                                     performers=PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=resolver, project_access_mode="acl",
                                   domain_auth=broker))
    login = client.post("/api/v1/auth/sessions", json={"username": "outage.user", "password": "pw"})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    fake.verify_error = LdapUnavailable("dc down")
    me = client.get("/api/v1/session", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert fake.calls == ["outage.user"]  # the authenticated path never touches AD


def test_health_reports_domain_auth_enabled(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({"e2e": _ldap_identity("e2e", display="E2E User", email="e2e@inme")})
    broker = make_broker(accounts, services.identities, fake, PostgresPerformerStore(services.engine))
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=HybridSessionResolver(accounts, services.identities, None,
                                                                          performers=PostgresPerformerStore(services.engine)),
                                   project_access_mode="acl", domain_auth=broker))
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["domain_auth"] is True


def test_health_reports_domain_auth_disabled(database):
    services, accounts, _owner, _config = database
    client = TestClient(create_app(services=services, account_store=accounts,
                                   session_resolver=HybridSessionResolver(accounts, services.identities, None,
                                                                          performers=PostgresPerformerStore(services.engine)),
                                   project_access_mode="acl"))
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["domain_auth"] is False
