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
                    "(account_id, username, username_key, display_name, password_hash, enabled, failed_attempts, created_at) "
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
