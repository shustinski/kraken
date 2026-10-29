"""Persistence for the administrator's deletion confirmation setting."""

from __future__ import annotations

from datetime import UTC, datetime

import sqlalchemy as sa

AUTO_CONFIRM_KEY = "auto_confirm_file_deletions"


def settings_table(metadata: sa.MetaData | None = None) -> sa.Table:
    return sa.Table(
        "admin_runtime_settings",
        metadata if metadata is not None else sa.MetaData(),
        sa.Column("setting_key", sa.Text, primary_key=True),
        sa.Column("setting_value", sa.Text, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def read_auto_confirm(connection, table) -> bool:
    value = connection.execute(sa.select(table.c.setting_value).where(
        table.c.setting_key == AUTO_CONFIRM_KEY,
    )).scalar()
    return str(value).lower() == "true"


def write_auto_confirm(connection, table, enabled: bool) -> None:
    now = datetime.now(UTC)
    stored = "true" if enabled else "false"
    existing = connection.execute(sa.select(table.c.setting_key).where(
        table.c.setting_key == AUTO_CONFIRM_KEY,
    )).first()
    if existing:
        connection.execute(table.update().where(
            table.c.setting_key == AUTO_CONFIRM_KEY,
        ).values(setting_value=stored, updated_at=now))
        return
    connection.execute(table.insert().values(
        setting_key=AUTO_CONFIRM_KEY, setting_value=stored, updated_at=now,
    ))


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
