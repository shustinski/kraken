"""Postgres account store: LDAP-backed accounts, verifier-based authenticate, session issuing."""

from __future__ import annotations

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_create_ldap_account_hash_is_unusable(database):
    services, accounts, _owner, _config = database
    account = accounts.create_ldap_account("ldap.user", "LDAP User")
    assert account.auth_provider == "ldap"
    assert accounts.authenticate("ldap.user", "anything") is None
    ok = accounts.authenticate("ldap.user", "anything", verifier=lambda stored, pw: True)
    assert ok is not None


def test_authenticate_verifier_counts_failures_and_blocks(database):
    services, accounts, _owner, _config = database
    accounts.create_ldap_account("lock.user", "Lock User")
    for _ in range(5):
        assert accounts.authenticate("lock.user", "x", verifier=lambda s, p: False) is None
    # 6th attempt is inside the lockout window even with a good verifier
    assert accounts.authenticate("lock.user", "x", verifier=lambda s, p: True) is None


def test_issue_session_respects_enabled(database):
    services, accounts, _owner, _config = database
    accounts.create_ldap_account("iss.user", "Iss User")
    assert accounts.issue_session("iss.user") is not None
    accounts.set_enabled(accounts.get_by_username("iss.user").account_id, False)
    assert accounts.issue_session("iss.user") is None


def test_record_audit_persists(database):
    services, accounts, _owner, _config = database
    accounts.record_audit(action="test.action", details={"k": "v"})
    rows = accounts.administration_audit(limit=10)
    assert any(r["action"] == "test.action" and r["details"] == {"k": "v"} for r in rows)


def test_convert_to_ldap_account(database):
    services, accounts, _o, _c = database
    account = accounts.create_account("conv.user", "Conv", "localpw")
    accounts.set_enabled(account.account_id, False)
    converted = accounts.convert_to_ldap_account(account.account_id)
    assert converted is not None and converted.auth_provider == "ldap" and converted.enabled is True
    assert accounts.authenticate("conv.user", "localpw") is None  # old local hash is gone
    assert accounts.convert_to_ldap_account("00000000-0000-0000-0000-000000000000") is None
