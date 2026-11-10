"""Shared fakes for the LDAP sign-in server tests (Tasks 6-9)."""

from __future__ import annotations

from kraken_server.composition import DomainAuthBroker


class FakeLdap:
    """Stand-in for LdapClient in server tests."""

    def __init__(self, identities: dict | None = None, *, verify_error: Exception | None = None):
        self.identities = identities or {}  # username -> LdapIdentity
        self.verify_error = verify_error
        self.calls: list[str] = []
        self.issuer = "ldap://fake:389"

    def verify(self, username, password):
        self.calls.append(username)
        if self.verify_error is not None:
            raise self.verify_error
        return self.identities.get(username)


def make_broker(accounts, identities, fake, performers):
    return DomainAuthBroker(accounts, identities, fake, performers=performers)
