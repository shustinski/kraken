"""LdapClient transport seam, filter escaping and typed error mapping (pure Python)."""

from __future__ import annotations

import pytest

from kraken_manager.infrastructure.auth.ldap import (
    DEFAULT_USER_FILTER,
    Ldap3Transport,
    LdapAuthenticationError,
    LdapClient,
    LdapIdentity,
    LdapUnavailable,
    resolve_ldap_tls,
)

_AMBIGUOUS = object()  # sentinel: transport returned >1 entry without raising


class FakeTransport:
    def __init__(self, *, service_ok=True, entries=(), user_bind_ok=True, error_on=None):
        self.service_ok = service_ok
        self.entries = entries
        self.user_bind_ok = user_bind_ok
        self.error_on = error_on  # "search" -> raise LdapUnavailable
        self.filters: list[str] = []
        self.user_binds: list[tuple[str, str]] = []
        self.closed = False

    def connect(self):
        if not self.service_ok:
            raise LdapAuthenticationError("service bind rejected")

    def search_user(self, f):
        self.filters.append(f)
        if self.error_on == "search":
            raise LdapUnavailable("ambiguous")
        return self.entries[0] if len(self.entries) == 1 else (None if not self.entries else _AMBIGUOUS)

    def user_bind(self, dn, password):
        self.user_binds.append((dn, password))
        return self.user_bind_ok

    def close(self):
        self.closed = True


def make_client(transport: FakeTransport) -> LdapClient:
    return LdapClient(
        "ldap://dc:389",
        base_dn="DC=ad,DC=inme",
        bind_dn="CN=kraken-svc,DC=ad,DC=inme",
        bind_password="svc-password",
        allow_insecure=True,
        transport=transport,
    )


_IVAN = (
    "CN=Ivan,DC=ad,DC=inme",
    {
        "sAMAccountName": "ivan",
        "displayName": "Ivan Petrov",
        "mail": "ivan@inme",
        "thumbnailPhoto": None,
    },
)


def test_resolve_ldap_tls_matrix() -> None:
    assert resolve_ldap_tls("ldaps://dc:636", start_tls=False, allow_insecure=False) == "ssl"
    assert resolve_ldap_tls("ldap://dc:389", start_tls=True, allow_insecure=False) == "starttls"
    assert resolve_ldap_tls("ldap://dc:389", start_tls=False, allow_insecure=True) == "plain"
    with pytest.raises(ValueError):
        resolve_ldap_tls("ldap://dc:389", start_tls=False, allow_insecure=False)
    with pytest.raises(ValueError):
        resolve_ldap_tls("http://x", start_tls=False, allow_insecure=False)


def test_client_constructor_rejects_plain_ldap_without_opt_in() -> None:
    with pytest.raises(ValueError):
        LdapClient(
            "ldap://dc:389",
            base_dn="DC=ad,DC=inme",
            bind_dn="CN=kraken-svc,DC=ad,DC=inme",
            bind_password="svc-password",
        )
    client = LdapClient(
        "ldap://dc:389",
        base_dn="DC=ad,DC=inme",
        bind_dn="CN=kraken-svc,DC=ad,DC=inme",
        bind_password="svc-password",
        start_tls=True,
    )
    assert client.issuer == "ldap://dc:389"


def test_verify_success_returns_identity() -> None:
    transport = FakeTransport(entries=(_IVAN,))
    identity = make_client(transport).verify("ivan", "secret")
    assert identity == LdapIdentity(username="ivan", display_name="Ivan Petrov", email="ivan@inme", avatar_url=None)
    assert transport.user_binds == [("CN=Ivan,DC=ad,DC=inme", "secret")]


def test_verify_wrong_password_returns_none() -> None:
    transport = FakeTransport(entries=(_IVAN,), user_bind_ok=False)
    assert make_client(transport).verify("ivan", "bad-password") is None
    assert transport.user_binds == [("CN=Ivan,DC=ad,DC=inme", "bad-password")]


def test_verify_unknown_user_returns_none() -> None:
    transport = FakeTransport(entries=())
    assert make_client(transport).verify("ghost", "pw") is None
    assert transport.user_binds == []
    assert transport.filters == [DEFAULT_USER_FILTER.format(username="ghost")]


def test_verify_ambiguous_search_raises_unavailable() -> None:
    other = ("CN=Ivan2,DC=ad,DC=inme", {"sAMAccountName": "ivan"})
    transport = FakeTransport(entries=(_IVAN, other))
    with pytest.raises(LdapUnavailable):
        make_client(transport).verify("ivan", "pw")
    assert transport.user_binds == []


def test_verify_service_bind_rejected_raises_authentication_error() -> None:
    transport = FakeTransport(service_ok=False)
    with pytest.raises(LdapAuthenticationError):
        make_client(transport).verify("ivan", "pw")
    assert transport.filters == []


def test_verify_transport_error_propagates_and_closes() -> None:
    transport = FakeTransport(entries=(_IVAN,), error_on="search")
    with pytest.raises(LdapUnavailable):
        make_client(transport).verify("ivan", "pw")
    assert transport.closed is True


def test_verify_escapes_username_in_filter() -> None:
    transport = FakeTransport(entries=(_IVAN,))
    make_client(transport).verify("a)(&*", "pw")
    assert transport.filters == [
        "(&(objectClass=user)(objectCategory=person)(sAMAccountName=a\\29\\28\\26\\2a))"
    ]


def test_invalid_email_becomes_none() -> None:
    entry = ("CN=Ivan,DC=ad,DC=inme", {"displayName": "Ivan Petrov", "mail": "not-an-email"})
    transport = FakeTransport(entries=(entry,))
    identity = make_client(transport).verify("ivan", "pw")
    assert identity is not None
    assert identity.email is None
    assert identity.display_name == "Ivan Petrov"


def test_display_name_falls_back_to_cn_then_username() -> None:
    cn_only = ("CN=Ivan,DC=ad,DC=inme", {"sAMAccountName": "ivan", "cn": "Ivan Petrovich"})
    transport = FakeTransport(entries=(cn_only,))
    identity = make_client(transport).verify("ivan", "pw")
    assert identity is not None
    assert identity.display_name == "Ivan Petrovich"
    bare = ("CN=Other,DC=ad,DC=inme", {})
    transport = FakeTransport(entries=(bare,))
    identity = make_client(transport).verify("someone", "pw")
    assert identity is not None
    assert identity.display_name == "someone"


def test_ldap3_transport_constructs_without_network() -> None:
    pytest.importorskip("ldap3")
    transport = Ldap3Transport(
        "ldap://dc:389",
        base_dn="DC=ad,DC=inme",
        bind_dn="CN=kraken-svc,DC=ad,DC=inme",
        bind_password="svc-password",
        start_tls=True,
        allow_insecure=False,
        ca_file=None,
        timeout=5.0,
    )
    assert transport.base_dn == "DC=ad,DC=inme"
    assert transport.timeout == 5.0


def test_ldap3_transport_rejects_plain_ldap_without_opt_in() -> None:
    pytest.importorskip("ldap3")
    with pytest.raises(ValueError):
        Ldap3Transport(
            "ldap://dc:389",
            base_dn="DC=ad,DC=inme",
            bind_dn="CN=kraken-svc,DC=ad,DC=inme",
            bind_password="svc-password",
            start_tls=False,
            allow_insecure=False,
            ca_file=None,
            timeout=5.0,
        )
