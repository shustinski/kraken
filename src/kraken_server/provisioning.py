"""Safe PostgreSQL provisioning used by the operator-facing CLI."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def validate_postgres_identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{label}: используйте не более 63 латинских букв, цифр и подчёркиваний; "
            "первый символ должен быть буквой или подчёркиванием"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class PostgresProvisioning:
    host: str
    port: int
    administrator: str
    administrator_password: str = field(repr=False)
    database: str
    application_user: str
    application_password: str = field(repr=False)
    maintenance_database: str = "postgres"

    @property
    def database_url(self) -> str:
        from sqlalchemy.engine import URL

        return URL.create(
            "postgresql+psycopg",
            username=self.application_user,
            password=self.application_password,
            host=self.host,
            port=self.port,
            database=self.database,
        ).render_as_string(hide_password=False)


def new_postgres_provisioning(
    *,
    host: str,
    port: int,
    administrator: str,
    administrator_password: str,
    database: str,
    application_user: str,
    application_password: str,
) -> PostgresProvisioning:
    if not host.strip():
        raise ValueError("Адрес PostgreSQL не может быть пустым")
    if not 1 <= int(port) <= 65535:
        raise ValueError("Порт PostgreSQL должен быть от 1 до 65535")
    if not administrator.strip():
        raise ValueError("Имя администратора PostgreSQL не может быть пустым")
    if not administrator_password:
        raise ValueError("Пароль администратора PostgreSQL не может быть пустым")
    if not application_password:
        raise ValueError("Пароль служебного пользователя БД не может быть пустым")
    return PostgresProvisioning(
        host=host.strip(),
        port=int(port),
        administrator=administrator.strip(),
        administrator_password=administrator_password,
        database=validate_postgres_identifier(database, "Имя базы данных"),
        application_user=validate_postgres_identifier(application_user, "Имя пользователя базы данных"),
        application_password=application_password,
    )


def _connector() -> Callable[..., Any]:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("The packaged PostgreSQL driver is missing") from exc
    return psycopg.connect


def provision_postgres(
    request: PostgresProvisioning,
    *,
    connect: Callable[..., Any] | None = None,
) -> None:
    """Create the dedicated login and database, failing before mutation on conflicts."""

    from psycopg import sql

    factory = connect or _connector()
    with factory(
        host=request.host,
        port=request.port,
        dbname=request.maintenance_database,
        user=request.administrator,
        password=request.administrator_password,
        autocommit=True,
    ) as connection:
        role_exists = connection.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (request.application_user,)
        ).fetchone()
        database_exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (request.database,)
        ).fetchone()
        conflicts = []
        if role_exists:
            conflicts.append(f"пользователь БД {request.application_user!r}")
        if database_exists:
            conflicts.append(f"база данных {request.database!r}")
        if conflicts:
            joined = " and ".join(conflicts)
            raise ValueError(
                f"Нельзя создать новый локальный сервер: {joined} уже существует. "
                "Выберите другие имена или используйте setup-server для существующей базы."
            )
        connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(request.application_user),
                sql.Literal(request.application_password),
            )
        )
        try:
            connection.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(request.database),
                    sql.Identifier(request.application_user),
                )
            )
        except BaseException:
            connection.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(request.application_user)))
            raise


def remove_provisioned_postgres(
    request: PostgresProvisioning,
    *,
    connect: Callable[..., Any] | None = None,
) -> None:
    """Roll back only objects created for the current failed initialization."""

    from psycopg import sql

    factory = connect or _connector()
    with factory(
        host=request.host,
        port=request.port,
        dbname=request.maintenance_database,
        user=request.administrator,
        password=request.administrator_password,
        autocommit=True,
    ) as connection:
        connection.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
            (request.database,),
        )
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(request.database)))
        connection.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(request.application_user)))


__all__ = [
    "PostgresProvisioning",
    "new_postgres_provisioning",
    "provision_postgres",
    "remove_provisioned_postgres",
    "validate_postgres_identifier",
]
