"""Confirmation required before any stored file is removed.

File removal is allowed only after the administrator types the project name,
or when that administrator has explicitly enabled auto-confirmation.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from kraken_server.services import ConflictError

AUTO_CONFIRM_KEY = "auto_confirm_file_deletions"


def settings_table(metadata: sa.MetaData | None = None) -> sa.Table:
    return sa.Table(
        "admin_runtime_settings",
        metadata if metadata is not None else sa.MetaData(),
        sa.Column("setting_key", sa.Text, primary_key=True),
        sa.Column("setting_value", sa.Text, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


class DeletionAuthorization:
    """Proof that this deletion attempt was confirmed or auto-confirmed."""

    __slots__ = ("_granted",)

    def __init__(self, granted: bool) -> None:
        self._granted = granted

    @classmethod
    def grant(
        cls,
        *,
        name_matches: bool,
        auto_confirm: bool,
        policy_enabled: bool,
    ) -> DeletionAuthorization:
        if auto_confirm and not policy_enabled:
            raise ConflictError("Автоподтверждение удалений выключено")
        if name_matches or auto_confirm:
            return cls(True)
        raise ConflictError("Удаление файлов требует подтверждения администратора")

    def require(self) -> None:
        if not self._granted:
            raise ConflictError("Удаление файлов требует подтверждения администратора")


def stage_stored_path(source: Path, trash: Path, authorization: DeletionAuthorization) -> None:
    authorization.require()
    trash.parent.mkdir(parents=True, exist_ok=True)
    source.rename(trash)


def remove_stored_path(path: Path, authorization: DeletionAuthorization) -> None:
    authorization.require()
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    shutil.rmtree(path)


class DeletionPolicy:
    """Shared administrator choice to skip interactive confirmation."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.table = settings_table()

    def enabled(self, connection=None) -> bool:
        if connection is None:
            with self.engine.connect() as owned:
                return self._read(owned)
        return self._read(connection)

    def _read(self, connection) -> bool:
        value = connection.execute(sa.select(self.table.c.setting_value).where(
            self.table.c.setting_key == AUTO_CONFIRM_KEY,
        )).scalar()
        return str(value).lower() == "true"

    def set_enabled(self, enabled: bool, accounts=None) -> None:
        now = datetime.now(UTC)
        stored = "true" if enabled else "false"
        with self.engine.begin() as connection:
            existing = connection.execute(sa.select(self.table.c.setting_key).where(
                self.table.c.setting_key == AUTO_CONFIRM_KEY,
            )).first()
            if existing:
                connection.execute(self.table.update().where(
                    self.table.c.setting_key == AUTO_CONFIRM_KEY,
                ).values(setting_value=stored, updated_at=now))
            else:
                connection.execute(self.table.insert().values(
                    setting_key=AUTO_CONFIRM_KEY, setting_value=stored, updated_at=now,
                ))
            if accounts is not None:
                action = "deletion.auto_confirm_enabled" if enabled else "deletion.auto_confirm_disabled"
                accounts._record_audit(connection, None, action, None, {"enabled": enabled})


def deletion_schema_ready(engine) -> bool:
    statements = (
        "SELECT reason FROM project_deletion_requests LIMIT 0",
        "SELECT setting_key FROM admin_runtime_settings LIMIT 0",
    )
    try:
        with engine.connect() as connection:
            for statement in statements:
                connection.execute(sa.text(statement))
    except (sa.exc.ProgrammingError, sa.exc.OperationalError):
        return False
    return True


def ensure_deletion_schema(engine) -> None:
    """Create the deletion tables when this database has not been migrated yet."""

    if deletion_schema_ready(engine):
        return
    if getattr(engine.dialect, "name", "") != "postgresql":
        raise RuntimeError("Заявки на удаление требуют PostgreSQL со схемой Kraken Server")
    from kraken_server.configuration import run_migrations

    run_migrations(engine.url.render_as_string(hide_password=False))
    if not deletion_schema_ready(engine):
        raise RuntimeError("После миграций таблица заявок на удаление всё ещё недоступна")


def short_database_error(exc: BaseException) -> str:
    text = str(exc).split("[SQL:", 1)[0].strip()
    return text or "Не удалось обратиться к базе"
