"""Empty project ACL is a successful empty role list, not a 404."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from kraken_manager.domain.identity import ProjectRole
from kraken_server.persistent_services_parts.discard_workspace_layer import DiscardWorkspaceLayerMixin
from kraken_server.services import NotFoundError


class _Identities:
    def __init__(self, people, roles=()):
        self.people = {str(person.id): person for person in people}
        self.roles = frozenset(roles)
        self.get_calls = 0

    def get(self, principal_id):
        self.get_calls += 1
        return self.people.get(str(principal_id)) if self.get_calls > 1 else None

    def list(self, *, include_inactive=False):
        del include_inactive
        return tuple(self.people.values())

    def roles_for(self, project_id, principal_id):
        del project_id, principal_id
        return self.roles


class _Events:
    def current_revision(self, stream_id):
        assert stream_id.startswith("acl:")
        return 0


class _Host(DiscardWorkspaceLayerMixin):
    def __init__(self, identities, accounts=None):
        self.identities = identities
        self.events = _Events()
        self.accounts = accounts

    def get_project(self, project_id):
        return {"project_id": project_id}


def test_known_principal_with_no_roles_returns_an_empty_list() -> None:
    principal_id = "06e05318-bdd7-4cd6-9bcd-a2d3b384c520"
    # get() misses the row; list() still has the catalogue entry.
    identities = _Identities([SimpleNamespace(id=principal_id)])
    result = _Host(identities).project_roles(str(uuid4()), principal_id)
    assert result["roles"] == []
    assert result["revision"] == 0
    assert result["principal_id"] == principal_id


def test_account_without_an_identity_row_has_an_empty_assignment() -> None:
    principal_id = str(uuid4())
    accounts = SimpleNamespace(get_account=lambda identifier: SimpleNamespace(account_id=identifier))
    result = _Host(_Identities([]), accounts).project_roles(str(uuid4()), principal_id)
    assert result["roles"] == []


def test_existing_roles_are_returned_when_the_identity_row_is_missing() -> None:
    identities = _Identities([], roles=(ProjectRole.VIEWER,))
    result = _Host(identities).project_roles(str(uuid4()), str(uuid4()))
    assert result["roles"] == ["viewer"]


def test_unknown_principal_is_still_not_found() -> None:
    principal_id = str(uuid4())
    with pytest.raises(NotFoundError) as caught:
        _Host(_Identities([])).project_roles(str(uuid4()), principal_id)
    assert principal_id in str(caught.value)


def test_empty_acl_route_returns_200() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kraken_server.app import create_app
    from kraken_server.services import InMemoryServerServices

    principal_id = "06e05318-bdd7-4cd6-9bcd-a2d3b384c520"
    services = InMemoryServerServices()
    services.identities = _Identities([SimpleNamespace(id=principal_id)])
    services.project_roles = lambda project_id, requested: _Host(
        services.identities,
    ).project_roles(project_id, requested)
    client = TestClient(create_app(services=services, development=True))
    created = client.post(
        "/api/v1/projects",
        headers={"Authorization": "Bearer developer", "Idempotency-Key": "empty-acl"},
        json={"name": "Participants", "width": 2, "height": 2, "orientation": "y_down"},
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]
    response = client.get(
        f"/api/v1/projects/{project_id}/acl/{principal_id}",
        headers={"Authorization": "Bearer developer"},
    )
    assert response.status_code == 200
    assert response.json()["roles"] == []
    missing = client.get(
        f"/api/v1/projects/{project_id}/acl/{uuid4()}",
        headers={"Authorization": "Bearer developer"},
    )
    assert missing.status_code == 404
    assert str(missing.json()["detail"])
