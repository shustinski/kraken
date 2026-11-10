"""DomainAuthBroker routing: JIT provisioning, AD-only re-verification, login claims."""

from __future__ import annotations

import pytest

from kraken_manager.domain.common import PrincipalId
from kraken_manager.domain.identity import Principal, PrincipalProvider
from kraken_manager.infrastructure.auth.ldap import LdapUnavailable
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_server.composition import HybridSessionResolver
from tests.ldap_fixtures import FakeLdap, make_broker

pytest_plugins = ["tests.test_admin_project_workflows"]


def _ldap_identity(username, display="Ivan Petrov", email=None, avatar=None):
    from kraken_manager.infrastructure.auth.ldap import LdapIdentity

    return LdapIdentity(username, display, email, avatar)


def test_jit_first_domain_login_creates_account_principal_performer(database):
    services, accounts, _owner, _config = database
    fake = FakeLdap({"ivan": _ldap_identity("ivan", email="ivan@inme")})
    performers = PostgresPerformerStore(services.engine)
    broker = make_broker(accounts, services.identities, fake, performers)
    session = broker.authenticate("ivan", "domain-pw")
    assert session is not None and session.account.auth_provider == "ldap"
    account = accounts.get_by_username("ivan")
    principal = services.identities.get(PrincipalId(str(account.account_id)))
    assert principal is None  # ldap principal id is uuid5, NOT the account id
    principal = services.identities.find_by_login("ivan")
    assert principal.provider is PrincipalProvider.LDAP
    assert principal.display_name == "Ivan Petrov" and principal.email == "ivan@inme"
    assert performers.get_by_principal(principal.id) is not None


def test_jit_wrong_password_creates_nothing(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, FakeLdap({}), None)
    assert broker.authenticate("ghost", "pw") is None
    assert accounts.get_by_username("ghost") is None


def test_broker_without_ldap_returns_none_for_unknown(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, None, None)
    assert broker.authenticate("anyone", "pw") is None


def test_local_account_never_touches_ldap(database):
    services, accounts, _o, _c = database
    fake = FakeLdap()
    broker = make_broker(accounts, services.identities, fake, None)
    accounts.create_account("local.only", "LO", "pw123")
    assert broker.authenticate("local.only", "pw123") is not None
    assert fake.calls == []


def test_ldap_account_reverified_each_login_and_ad_down_propagates(database):
    services, accounts, _o, _c = database
    fake = FakeLdap({"re.user": _ldap_identity("re.user")})
    broker = make_broker(accounts, services.identities, fake, None)
    assert broker.authenticate("re.user", "ok") is not None
    fake.verify_error = LdapUnavailable("dc down")
    with pytest.raises(LdapUnavailable):
        broker.authenticate("re.user", "ok")


def test_login_conflict_transfers_claim_from_squatting_local_account(database):
    services, accounts, _o, _c = database
    # local account "squatter" already logged in once -> principal holds the login claim
    account = accounts.create_account("squatter", "Squatter", "pw")
    local_pid = PrincipalId(str(account.account_id))
    services.identities.save(Principal.local(subject="squatter", display_name="Squatter",
                                             principal_id=local_pid, login="squatter"))
    broker = make_broker(accounts, services.identities, FakeLdap({"squatter": _ldap_identity("squatter")}), None)
    session = broker.authenticate("squatter", "domain-pw")
    assert session is not None
    # the row is converted for the domain user (the local identity is what stops working)
    converted = accounts.get_by_username("squatter")
    assert converted.auth_provider == "ldap" and converted.enabled is True
    assert accounts.authenticate("squatter", "pw") is None  # local password no longer works
    # claim released from the local principal, held by the LDAP principal
    assert services.identities.get(local_pid).login is None
    assert services.identities.find_by_login("squatter").provider is PrincipalProvider.LDAP
    assert any(r["action"] == "login_conflict_local_disabled" for r in accounts.administration_audit(limit=20))
    # the old local user is out: a fresh local sign-in is impossible, the domain session works
    assert session.account.account_id == converted.account_id


def test_domain_login_shares_row_with_never_signed_in_local_account(database):
    services, accounts, _o, _c = database
    # local account exists but its user never signed in: no principal, no claim
    accounts.create_account("shared.user", "Shared Local", "localpw")
    fake = FakeLdap({"shared.user": _ldap_identity("shared.user")})
    broker = make_broker(accounts, services.identities, fake, None)
    session = broker.authenticate("shared.user", "domain-pw")
    assert session is not None
    # the row stays LOCAL and enabled — the local user keeps working (no lockout in either direction)
    row = accounts.get_by_username("shared.user")
    assert row.auth_provider == "local" and row.enabled is True
    assert accounts.authenticate("shared.user", "localpw") is not None
    assert fake.calls == ["shared.user"]  # local sign-in above did not touch AD
    # the LDAP principal holds the claim
    assert services.identities.find_by_login("shared.user").provider is PrincipalProvider.LDAP
    # the local user's first-login provisioning yields the claim
    resolver = HybridSessionResolver(accounts, services.identities, None)
    local_session = accounts.authenticate("shared.user", "localpw")
    resolver.provision_local_account(local_session.account)
    local_principal = services.identities.get(PrincipalId(local_session.account.account_id))
    assert local_principal.login is None
    assert any(r["action"] == "login_claim_conflict_local_skipped" for r in accounts.administration_audit(limit=20))


def test_reverse_interleave_local_login_keeps_working(database):
    services, accounts, _o, _c = database
    from uuid import NAMESPACE_URL, uuid5

    ldap_pid = str(uuid5(NAMESPACE_URL, "kraken:ldap:late.user"))
    services.identities.save(Principal.ldap(issuer="ldap://fake:389", subject="late.user",
                                            display_name="Late User", principal_id=ldap_pid, login="late.user"))
    accounts.create_account("late.user", "Late Local", "pw")
    resolver = HybridSessionResolver(accounts, services.identities, None)
    session = accounts.authenticate("late.user", "pw")
    assert session is not None
    resolver.provision_local_account(session.account)
    local_principal = services.identities.get(PrincipalId(session.account.account_id))
    assert local_principal.login is None  # claim yielded to the LDAP principal
    assert any(r["action"] == "login_claim_conflict_local_skipped" for r in accounts.administration_audit(limit=20))


def test_disabled_ldap_account_cannot_login(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, FakeLdap({"off.user": _ldap_identity("off.user")}), None)
    broker.authenticate("off.user", "pw")
    accounts.set_enabled(accounts.get_by_username("off.user").account_id, False)
    assert broker.authenticate("off.user", "pw") is None


def test_second_jit_for_same_domain_user_is_idempotent(database):
    services, accounts, _o, _c = database
    broker = make_broker(accounts, services.identities, FakeLdap({"twice": _ldap_identity("twice")}), None)
    first = broker.authenticate("twice", "pw")
    second = broker.authenticate("twice", "pw")
    assert first is not None and second is not None
    assert accounts.get_by_username("twice").account_id == first.account.account_id
