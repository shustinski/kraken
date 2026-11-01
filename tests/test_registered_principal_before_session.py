"""A registered server account is a principal before its first API call.

Reproduces SRV-ROLE-BEFORE-SESSION: the project owner must be able to grant a
role immediately after POST /auth/accounts, without the new account calling
the API.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_manager.domain.common import PrincipalId, ProjectId
from kraken_manager.infrastructure.postgres.identity_store import PostgresIdentityAclStore
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_server.app import create_app
from kraken_server.composition import HybridSessionResolver
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


class _TokenLog:
    def __init__(self, resolver: HybridSessionResolver) -> None:
        self._resolver = resolver
        self.tokens: list[str] = []

    def __call__(self, token: str):
        self.tokens.append(token)
        return self._resolver(token)

    def __getattr__(self, name: str):
        return getattr(self._resolver, name)


def test_owner_grants_viewer_before_the_new_account_calls_the_api(database) -> None:
    services, accounts, owner, _config = database
    services.accounts = accounts
    created = project(services, owner)
    performers = PostgresPerformerStore(services.engine)
    resolver = HybridSessionResolver(accounts, services.identities, None, performers=performers)
    observed = _TokenLog(resolver)
    client = TestClient(
        create_app(
            services=services,
            account_store=accounts,
            session_resolver=observed,
            project_access_mode="acl",
        )
    )
    owner_session = accounts.authenticate("admin", "password")
    assert owner_session is not None

    registered = client.post(
        "/api/v1/auth/accounts",
        json={"username": "new-viewer", "password": "secret", "display_name": "New Viewer"},
    )
    assert registered.status_code == 201, registered.text
    body = registered.json()
    principal_id = body["principal"]["id"]
    new_token = body["access_token"]
    assert observed.tokens == []
    persisted = PostgresIdentityAclStore(services.engine).get(PrincipalId(principal_id))
    assert persisted is not None
    assert persisted.system_roles == frozenset()
    assert services.identities.roles_for(ProjectId(created["project_id"]), PrincipalId(principal_id)) == frozenset()

    assigned = client.put(
        f"/api/v1/projects/{created['project_id']}/acl/{principal_id}/viewer",
        headers={
            "Authorization": f"Bearer {owner_session.token}",
            "Idempotency-Key": "grant-viewer-before-session",
            "If-Match": "0",
        },
    )
    assert assigned.status_code == 200, assigned.text
    assert assigned.json()["roles"] == ["viewer"]
    assert observed.tokens == [owner_session.token]

    persisted = PostgresIdentityAclStore(services.engine).get(PrincipalId(principal_id))
    assert persisted is not None
    assert persisted.subject == "new-viewer"
    assert persisted.active is True
    linked = PostgresPerformerStore(services.engine).get_by_principal(PrincipalId(principal_id))
    assert linked is not None
    assert linked.name == "New Viewer"
    assert linked.active is True

    visible = client.get(
        f"/api/v1/projects/{created['project_id']}",
        headers={"Authorization": f"Bearer {new_token}"},
    )
    assert visible.status_code == 200, visible.text
    assert visible.json()["name"] == created["name"]

    repeated = client.post(
        "/api/v1/auth/accounts",
        json={"username": "new-viewer", "password": "secret", "display_name": "New Viewer"},
    )
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["principal"]["id"] == principal_id
    assert services.project_roles(created["project_id"], principal_id)["roles"] == ["viewer"]
    assert PostgresIdentityAclStore(services.engine).get(PrincipalId(principal_id)) is not None
    assert PostgresPerformerStore(services.engine).get_by_principal(PrincipalId(principal_id)) is not None


def test_same_password_retry_materializes_a_principal_the_first_request_missed(database) -> None:
    services, accounts, _owner, _config = database
    resolver = HybridSessionResolver(
        accounts, services.identities, None, performers=PostgresPerformerStore(services.engine)
    )
    observed = _TokenLog(resolver)
    client = TestClient(
        create_app(
            services=services,
            account_store=accounts,
            session_resolver=observed,
            project_access_mode="acl",
        )
    )
    account = accounts.create_account("retry-viewer", "Retry Viewer", "secret")
    assert services.identities.get(PrincipalId(account.account_id)) is None

    registered = client.post(
        "/api/v1/auth/accounts",
        json={"username": "retry-viewer", "password": "secret", "display_name": "Retry Viewer"},
    )
    assert registered.status_code == 201, registered.text
    assert registered.json()["principal"]["id"] == account.account_id
    assert observed.tokens == []

    persisted = PostgresIdentityAclStore(services.engine).get(PrincipalId(account.account_id))
    assert persisted is not None
    assert persisted.subject == "retry-viewer"
    assert persisted.active is True
    assert persisted.system_roles == frozenset()
    linked = PostgresPerformerStore(services.engine).get_by_principal(PrincipalId(account.account_id))
    assert linked is not None
    assert linked.name == "Retry Viewer"
