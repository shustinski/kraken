from __future__ import annotations

import pytest

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.identity import Principal, PrincipalProvider, SystemRole


def test_provider_enum_exposes_ldap() -> None:
    assert PrincipalProvider.LDAP.value == "ldap"


def test_ldap_principal_requires_issuer() -> None:
    # Factory level: issuer is a required keyword-only argument (like the gitlab factory).
    with pytest.raises(TypeError):
        Principal.ldap(subject="ivan", display_name="Ivan")
    # Domain level: an LDAP principal constructed without an issuer is rejected.
    with pytest.raises(DomainValidationError):
        Principal(
            id="11111111-2222-3333-4444-555555555555",
            provider=PrincipalProvider.LDAP,
            subject="ivan",
            display_name="Ivan",
        )


def test_ldap_external_key_is_casefolded_login() -> None:
    principal = Principal.ldap(issuer="ldap://dc:389", subject="Ivan.P", display_name="Ivan")
    assert principal.provider is PrincipalProvider.LDAP
    assert principal.external_key == "ldap:ivan.p"
    assert principal.login == "Ivan.P"
    assert principal.login_key == "ivan.p"


def test_ldap_login_defaults_to_subject_and_explicit_login_wins() -> None:
    principal = Principal.ldap(issuer="ldap://dc:389", subject="Subj", display_name="Ivan")
    assert principal.login == "Subj"
    assert principal.login_key == "subj"
    explicit = Principal.ldap(
        issuer="ldap://dc:389", subject="Subj", display_name="Ivan", login="Explicit.Login"
    )
    assert explicit.login == "Explicit.Login"
    assert explicit.login_key == "explicit.login"


def test_profile_fields_round_trip() -> None:
    principal = Principal.ldap(
        issuer="ldap://dc:389",
        subject="ivan",
        display_name="Ivan",
        email="ivan@example.test",
        login="ivan",
        avatar_url="https://x/y.png",
        avatar_source="ldap",
        status_text="в работе",
        system_roles=frozenset({SystemRole.SERVER_ADMIN}),
    )
    assert principal.login_key == "ivan"
    assert principal.avatar_url == "https://x/y.png"
    assert principal.avatar_source == "ldap"
    assert principal.status_text == "в работе"
    assert principal.email == "ivan@example.test"
    assert principal.system_roles == frozenset({SystemRole.SERVER_ADMIN})


def test_ldap_principal_without_login_has_no_login_key() -> None:
    principal = Principal(
        id="11111111-2222-3333-4444-555555555555",
        provider=PrincipalProvider.LDAP,
        issuer="ldap://dc:389",
        subject="ivan",
        display_name="Ivan",
    )
    assert principal.login is None
    assert principal.login_key is None
    assert principal.external_key == "ldap:ivan"


def test_status_text_must_be_at_most_255_chars() -> None:
    with pytest.raises(DomainValidationError):
        Principal.ldap(issuer="ldap://dc:389", subject="ivan", display_name="Ivan", status_text="x" * 256)
    principal = Principal.ldap(
        issuer="ldap://dc:389", subject="ivan", display_name="Ivan", status_text="x" * 255
    )
    assert principal.status_text == "x" * 255


def test_status_text_empty_string_allowed() -> None:
    principal = Principal.ldap(issuer="ldap://dc:389", subject="ivan", display_name="Ivan", status_text="")
    assert principal.status_text == ""


def test_login_must_be_at_most_128_chars() -> None:
    with pytest.raises(DomainValidationError):
        Principal.ldap(issuer="ldap://dc:389", subject="ivan", display_name="Ivan", login="x" * 129)


def test_login_whitespace_only_rejected() -> None:
    with pytest.raises(DomainValidationError):
        Principal.ldap(issuer="ldap://dc:389", subject="ivan", display_name="Ivan", login="   ")


def test_avatar_url_must_be_at_most_2048_chars() -> None:
    with pytest.raises(DomainValidationError):
        Principal.ldap(
            issuer="ldap://dc:389", subject="ivan", display_name="Ivan", avatar_url="https://x/" + "y" * 2044
        )


def test_avatar_source_must_be_ldap_or_manual() -> None:
    with pytest.raises(DomainValidationError):
        Principal.ldap(
            issuer="ldap://dc:389", subject="ivan", display_name="Ivan", avatar_source="bogus"
        )


def test_local_factory_accepts_profile_kwargs() -> None:
    principal = Principal.local(
        subject="ivan",
        display_name="Ivan",
        login="ivan",
        avatar_url="https://x/y.png",
        avatar_source="manual",
        status_text="ok",
    )
    assert principal.provider is PrincipalProvider.LOCAL
    assert principal.external_key == "local:ivan"
    assert principal.login == "ivan"
    assert principal.login_key == "ivan"
    assert principal.avatar_url == "https://x/y.png"
    assert principal.avatar_source == "manual"
    assert principal.status_text == "ok"


def test_local_factory_profile_kwargs_validate_like_ldap() -> None:
    with pytest.raises(DomainValidationError):
        Principal.local(subject="ivan", display_name="Ivan", avatar_source="bogus")
    with pytest.raises(DomainValidationError):
        Principal.local(subject="ivan", display_name="Ivan", status_text="x" * 256)
    with pytest.raises(DomainValidationError):
        Principal.local(subject="ivan", display_name="Ivan", login="x" * 129)


def test_local_principal_without_login_has_no_login_key() -> None:
    principal = Principal.local(subject="ivan", display_name="Ivan")
    assert principal.login is None
    assert principal.login_key is None
    assert principal.external_key == "local:ivan"


def test_gitlab_external_key_and_fields_unchanged() -> None:
    principal = Principal.gitlab(issuer="https://gitlab.example", subject="42", display_name="Ivan")
    assert principal.external_key == "gitlab:https://gitlab.example:42"
    assert principal.login is None
    assert principal.login_key is None


def test_local_principal_still_rejects_issuer() -> None:
    with pytest.raises(DomainValidationError):
        Principal(
            id="11111111-2222-3333-4444-555555555555",
            provider=PrincipalProvider.LOCAL,
            issuer="ldap://dc:389",
            subject="ivan",
            display_name="Ivan",
        )
