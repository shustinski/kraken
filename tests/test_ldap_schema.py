"""Schema coverage for the LDAP sign-in migration: logins, profiles, auth_provider."""
from __future__ import annotations

import os
from uuid import uuid4

import pytest
import sqlalchemy as sa

from kraken_manager.domain.identity import Principal

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_new_account_defaults_to_local_provider(database):
    services, accounts, _owner, _config = database
    account = accounts.create_account("prov-check", "Prov Check", "pw")
    assert accounts.get_account(account.account_id).auth_provider == "local"


def test_login_key_unique_index_rejects_duplicates(database):
    services, accounts, _owner, _config = database
    a = Principal.local(
        subject="dup-user",
        display_name="A",
        principal_id="11111111-1111-1111-1111-111111111111",
        login="dup-user",
    )
    services.identities.save(a)
    b = Principal.local(
        subject="dup-user-2",
        display_name="B",
        principal_id="22222222-2222-2222-2222-222222222222",
        login="DUP-USER",
    )
    with pytest.raises(sa.exc.IntegrityError):
        services.identities.save(b)


# Exact statements from migration 20261110_0012 (pinned here as the contract under test).
_BACKFILL_SQL = """
    UPDATE principals p
       SET login = a.username, login_key = a.username_key
     FROM server_accounts a
    WHERE p.provider = 'local'
      AND p.external_key = 'local:' || a.username
      AND p.login_key IS NULL
      AND NOT EXISTS (
          SELECT 1 FROM principals q
           WHERE q.login_key = a.username_key AND q.principal_id <> p.principal_id)
"""
_AUDIT_SQL = """
    INSERT INTO administration_audit (actor_id, action, target_account_id, details, recorded_at)
    SELECT NULL, 'login_claim_conflict_backfill', p.principal_id,
           jsonb_build_object('login', a.username), now()
      FROM server_accounts a
      JOIN principals p ON p.external_key = 'local:' || a.username
     WHERE p.login_key IS NULL
       AND EXISTS (SELECT 1 FROM principals q
                    WHERE q.login_key = a.username_key AND q.principal_id <> p.principal_id)
"""


def test_migration_backfills_local_principal_logins(tmp_path, monkeypatch):
    url = os.environ.get("KRAKEN_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("KRAKEN_TEST_POSTGRES_URL is not configured")
    from alembic import command
    from alembic.config import Config

    base = sa.create_engine(url)
    schema = "ldap_backfill_" + uuid4().hex
    with base.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    scoped = sa.engine.make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    monkeypatch.setenv("KRAKEN_DATABASE_URL", scoped.render_as_string(hide_password=False))
    cfg = Config("alembic.ini")
    engine = sa.create_engine(scoped)
    try:
        command.upgrade(cfg, "20261029_0011")
        with engine.begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO server_accounts "
                    "(account_id, username, username_key, display_name, password_hash, "
                    "enabled, failed_attempts, created_at) "
                    "VALUES (:account_id, 'backfill.user', 'backfill.user', 'BF', 'x', true, 0, now())"
                ),
                {"account_id": str(uuid4())},
            )
            connection.execute(
                sa.text(
                    "INSERT INTO principals "
                    "(principal_id, external_key, provider, subject, display_name, enabled, system_roles) "
                    "SELECT account_id, 'local:backfill.user', 'local', 'backfill.user', 'BF', true, '[]' "
                    "FROM server_accounts WHERE username='backfill.user'"
                )
            )
        command.upgrade(cfg, "head")
        with engine.connect() as connection:
            row = connection.execute(
                sa.text("SELECT login, login_key FROM principals WHERE external_key='local:backfill.user'")
            ).mappings().one()
        assert row["login"] == "backfill.user"
        assert row["login_key"] == "backfill.user"
    finally:
        engine.dispose()
        with base.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
        base.dispose()


def test_migration_backfill_skips_claimed_logins_and_writes_audit(tmp_path, monkeypatch):
    """Conflict safety net: a claimed login is skipped and gets an audit row.

    A claim cannot be seeded before the head upgrade — the old schema has no
    ``login_key`` column — so the conflict state is seeded right after the
    upgrade and the migration's exact backfill/audit statements are re-run
    against it.
    """
    url = os.environ.get("KRAKEN_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("KRAKEN_TEST_POSTGRES_URL is not configured")
    from alembic import command
    from alembic.config import Config

    base = sa.create_engine(url)
    schema = "ldap_conflict_" + uuid4().hex
    with base.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    scoped = sa.engine.make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    monkeypatch.setenv("KRAKEN_DATABASE_URL", scoped.render_as_string(hide_password=False))
    cfg = Config("alembic.ini")
    engine = sa.create_engine(scoped)
    try:
        command.upgrade(cfg, "20261029_0011")
        command.upgrade(cfg, "head")
        local_pid = str(uuid4())
        with engine.begin() as connection:
            connection.execute(
                sa.text(
                    "INSERT INTO server_accounts "
                    "(account_id, username, username_key, display_name, password_hash, "
                    "enabled, failed_attempts, created_at) "
                    "VALUES (:account_id, 'claimed.user', 'claimed.user', 'CL', 'x', true, 0, now())"
                ),
                {"account_id": str(uuid4())},
            )
            connection.execute(
                sa.text(
                    "INSERT INTO principals "
                    "(principal_id, external_key, provider, issuer, subject, display_name, enabled, system_roles, "
                    "login, login_key) "
                    "VALUES (:p, 'ldap:claimed.user', 'ldap', 'ldap://dc.test:389', 'claimed.user', 'CL', true, '[]', "
                    "'claimed.user', 'claimed.user')"
                ),
                {"p": str(uuid4())},
            )
            connection.execute(
                sa.text(
                    "INSERT INTO principals "
                    "(principal_id, external_key, provider, subject, display_name, enabled, system_roles) "
                    "VALUES (:p, 'local:claimed.user', 'local', 'claimed.user', 'CL', true, '[]')"
                ),
                {"p": local_pid},
            )
            connection.execute(sa.text(_BACKFILL_SQL))
            connection.execute(sa.text(_AUDIT_SQL))
        with engine.connect() as connection:
            row = connection.execute(
                sa.text("SELECT login, login_key FROM principals WHERE external_key='local:claimed.user'")
            ).mappings().one()
            audit = connection.execute(
                sa.text(
                    "SELECT action, target_account_id, details FROM administration_audit "
                    "WHERE action='login_claim_conflict_backfill'"
                )
            ).mappings().all()
        assert row["login"] is None
        assert row["login_key"] is None
        assert len(audit) == 1
        assert str(audit[0]["target_account_id"]) == local_pid
        assert audit[0]["details"] == {"login": "claimed.user"}
    finally:
        engine.dispose()
        with base.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA "{schema}" CASCADE'))
        base.dispose()
