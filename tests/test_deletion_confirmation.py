from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa

from kraken_admin.deletion_confirmation import (
    DeletionAuthorization,
    DeletionPolicy,
    deletion_schema_ready,
    ensure_deletion_schema,
    remove_stored_path,
    settings_table,
    short_database_error,
)
from kraken_server.project_deletion import DeletionRequests, normalize_deletion_reason, request_table
from kraken_server.services import ConflictError, ValidationError


def test_deletion_request_accepts_missing_body() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kraken_server.app import create_app

    app = create_app(development=True)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/projects/11111111-1111-1111-1111-111111111111/deletion-requests",
            headers={"Authorization": "Bearer dev"},
        )
    assert response.status_code == 409


def test_deletion_reason_is_optional_and_limited() -> None:
    assert normalize_deletion_reason(None) == ""
    assert normalize_deletion_reason("  old   scan  ") == "old scan"
    with pytest.raises(ValidationError, match="текстом"):
        normalize_deletion_reason(12)
    with pytest.raises(ValidationError, match="500"):
        normalize_deletion_reason("x" * 501)


def test_file_removal_requires_confirmation(tmp_path: Path) -> None:
    target = tmp_path / "frame.jpg"
    target.write_bytes(b"image")
    denied = DeletionAuthorization(False)
    with pytest.raises(ConflictError, match="подтверждения"):
        remove_stored_path(target, denied)
    assert target.exists()
    with pytest.raises(ConflictError, match="подтверждения"):
        DeletionAuthorization.grant(name_matches=False, auto_confirm=False, policy_enabled=False)
    with pytest.raises(ConflictError, match="выключено"):
        DeletionAuthorization.grant(name_matches=False, auto_confirm=True, policy_enabled=False)
    remove_stored_path(
        target,
        DeletionAuthorization.grant(name_matches=False, auto_confirm=True, policy_enabled=True),
    )
    assert not target.exists()


def test_reason_and_auto_confirm_setting_are_durable(tmp_path: Path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'admin.sqlite3'}")
    request_table().create(engine)
    settings_table().create(engine)
    requests = DeletionRequests(engine)
    project_id, actor_id = str(uuid4()), str(uuid4())
    created = requests.request(project_id, "Example", actor_id, "  duplicate  ")
    assert created["reason"] == "duplicate"
    again = requests.request(project_id, "Example", actor_id, "")
    assert again["request_id"] == created["request_id"]
    assert again["reason"] == "duplicate"
    updated = requests.request(project_id, "Example", actor_id, "replaced")
    assert updated["reason"] == "replaced"

    policy = DeletionPolicy(engine)
    assert policy.enabled() is False
    recorded: list[str] = []

    class _Accounts:
        def _record_audit(self, connection, actor_id, action, target, details) -> None:
            del connection, actor_id, target, details
            recorded.append(action)

    policy.set_enabled(True, _Accounts())
    assert policy.enabled() is True
    assert recorded == ["deletion.auto_confirm_enabled"]
    policy.set_enabled(False, _Accounts())
    assert policy.enabled() is False
    engine.dispose()


def test_missing_deletion_table_is_created_by_migration(tmp_path: Path, monkeypatch) -> None:
    inner = sa.create_engine(f"sqlite:///{tmp_path / 'missing.sqlite3'}")
    assert deletion_schema_ready(inner) is False

    class _Engine:
        dialect = inner.dialect
        url = inner.url

        def connect(self):
            return inner.connect()

    _Engine.dialect = type("Dialect", (), {"name": "postgresql"})()
    migrated: list[str] = []

    def migrate(url: str) -> None:
        migrated.append(url)
        request_table().create(inner)
        settings_table().create(inner)

    monkeypatch.setattr("kraken_server.configuration.run_migrations", migrate)
    ensure_deletion_schema(_Engine())
    assert migrated == [inner.url.render_as_string(hide_password=False)]
    assert deletion_schema_ready(inner) is True
    ensure_deletion_schema(_Engine())
    assert len(migrated) == 1
    inner.dispose()


def test_database_error_hides_the_sql_statement() -> None:
    message = short_database_error(
        RuntimeError(
            '(psycopg.errors.UndefinedTable) отношение "project_deletion_requests" не существует\n'
            "[SQL: SELECT project_deletion_requests.request_id FROM project_deletion_requests]"
        )
    )
    assert "project_deletion_requests" in message
    assert "[SQL:" not in message
