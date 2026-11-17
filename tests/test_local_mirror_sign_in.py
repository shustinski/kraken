"""A remotely verified sign-in completes locally through a mirrored account.

``login_or_create_local`` signs in the workstation-local account for a
username already verified against Kraken Server (or AD), creating it when
absent. An existing local account's password is never overwritten.
"""
from __future__ import annotations

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService


def _service_with_owner(tmp_path) -> EmbeddedProjectService:
    service = EmbeddedProjectService(tmp_path / "local")
    service.create_initial_account("owner", "Owner", "secret")
    return service


def test_mirror_account_created_and_sign_in_works(tmp_path) -> None:
    service = _service_with_owner(tmp_path)
    session = service.login_or_create_local("alice", "pw")
    assert session is not None
    assert session.principal.subject == "alice"
    assert service.login("alice", "pw") is not None
    assert service.accounts.account_count() == 2


def test_existing_local_account_with_matching_password_signs_in_without_duplicate(tmp_path) -> None:
    service = _service_with_owner(tmp_path)
    session = service.login_or_create_local("OWNER", "secret")
    assert session is not None
    assert service.accounts.account_count() == 1


def test_existing_local_account_with_other_password_returns_none(tmp_path) -> None:
    service = _service_with_owner(tmp_path)
    assert service.login_or_create_local("owner", "other") is None
    assert service.login("owner", "secret") is not None
    assert service.accounts.account_count() == 1


def test_invalid_username_returns_none(tmp_path) -> None:
    service = _service_with_owner(tmp_path)
    assert service.login_or_create_local("bad name", "pw") is None
    assert service.accounts.account_count() == 1


def test_catalog_delegates_to_local(tmp_path) -> None:
    local = _service_with_owner(tmp_path)
    catalog = DualCatalogService(local)
    session = catalog.login_or_create_local("alice", "pw")
    assert session is not None
    assert session.principal.subject == "alice"
