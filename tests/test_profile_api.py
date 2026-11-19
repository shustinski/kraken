"""Own-profile PATCH endpoint and profile fields in the payloads.

Task 9: a user updates their own status text and manual avatar through the
identity store only (no AD calls on any authenticated path), and the updated
profile is visible in GET /api/v1/session and GET /api/v1/principals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import sqlalchemy as sa

from kraken_manager.domain.identity import Principal
from kraken_server.app import create_app

pytest_plugins = ["tests.test_admin_project_workflows"]


def _app(database, *, with_identities: bool = True) -> TestClient:
    services, accounts, _person, _config = database
    kwargs = {"services": services, "account_store": accounts, "project_access_mode": "acl"}
    if with_identities:
        kwargs["identities"] = services.identities
    return TestClient(create_app(**kwargs))


def _admin_headers(database) -> dict[str, str]:
    _services, accounts, _person, _config = database
    session = accounts.authenticate("admin", "password")
    assert session is not None
    return {"Authorization": f"Bearer {session.token}"}


def test_patch_own_profile_is_visible_in_session_and_principals(database):
    services, accounts, person, _config = database
    # The fixture's legacy admin row predates the login claim (Task 3): mirror
    # the production provisioning path, which claims the local account's login.
    services.identities.save(
        Principal.local(subject="admin", display_name="Administrator", principal_id=person.id, login="admin")
    )
    client = _app(database)
    headers = _admin_headers(database)

    patched = client.patch(
        "/api/v1/principals/me/profile",
        json={"status_text": "на встрече", "avatar_url": "https://me/1.png"},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json() == {
        "login": "admin",
        "display_name": "Administrator",
        "avatar_url": "https://me/1.png",
        "status_text": "на встрече",
    }

    me = client.get("/api/v1/session", headers=headers)
    assert me.status_code == 200, me.text
    assert me.json()["status_text"] == "на встрече"
    assert me.json()["avatar_url"] == "https://me/1.png"

    principals = client.get("/api/v1/principals", headers=headers)
    assert principals.status_code == 200, principals.text
    admin = next(item for item in principals.json()["items"] if item["principal_id"] == str(person.id))
    assert admin["login"] == "admin"
    assert admin["status_text"] == "на встрече"
    assert admin["avatar_url"] == "https://me/1.png"


def test_patch_validates_lengths(database):
    client = _app(database)
    headers = _admin_headers(database)

    too_long_status = client.patch(
        "/api/v1/principals/me/profile", json={"status_text": "x" * 256}, headers=headers
    )
    assert too_long_status.status_code == 422, too_long_status.text
    too_long_avatar = client.patch(
        "/api/v1/principals/me/profile", json={"avatar_url": "x" * 2049}, headers=headers
    )
    assert too_long_avatar.status_code == 422, too_long_avatar.text
    empty_avatar = client.patch("/api/v1/principals/me/profile", json={"avatar_url": ""}, headers=headers)
    assert empty_avatar.status_code == 422, empty_avatar.text


def test_manual_avatar_survives_ldap_sync(database):
    services, accounts, person, _config = database
    client = _app(database)
    headers = _admin_headers(database)

    patched = client.patch(
        "/api/v1/principals/me/profile", json={"avatar_url": "https://me/manual.png"}, headers=headers
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["avatar_url"] == "https://me/manual.png"

    services.identities.sync_avatar(person.id, "https://ad/x.png")
    updated = services.identities.get(person.id)
    assert updated is not None
    assert updated.avatar_url == "https://me/manual.png"
    assert updated.avatar_source == "manual"


def test_new_local_registration_payload_has_profile_fields(database):
    services, accounts, _person, _config = database
    services.accounts = accounts  # list_principals merges accounts without a principal row
    client = _app(database)

    created = client.post("/api/v1/auth/accounts", json={"username": "profile-user", "password": "secret"})
    assert created.status_code == 201, created.text
    token = created.json()["access_token"]

    principals = client.get("/api/v1/principals", headers={"Authorization": f"Bearer {token}"})
    assert principals.status_code == 200, principals.text
    item = next(
        entry
        for entry in principals.json()["items"]
        if entry["principal_id"] == created.json()["principal"]["id"]
    )
    assert item["login"] == "profile-user"
    assert item["avatar_url"] is None
    assert item["status_text"] is None


def test_patch_profile_reports_missing_identity_catalog(database):
    client = _app(database, with_identities=False)
    headers = _admin_headers(database)
    response = client.patch(
        "/api/v1/principals/me/profile", json={"status_text": "на встрече"}, headers=headers
    )
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "Identity catalog is not configured"


def test_patch_profile_returns_404_when_the_principal_row_is_missing(database):
    services, accounts, person, _config = database
    client = _app(database)
    headers = _admin_headers(database)
    with services.engine.begin() as connection:
        connection.execute(
            sa.text("DELETE FROM principals WHERE principal_id = :pid"), {"pid": str(person.id)}
        )
    response = client.patch(
        "/api/v1/principals/me/profile", json={"status_text": "на встрече"}, headers=headers
    )
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Profile principal row is missing"
