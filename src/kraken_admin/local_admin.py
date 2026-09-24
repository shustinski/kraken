"""Local server administration. These operations never go through HTTP."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from kraken_manager.application.acl import RevokeProjectRoleHandler
from kraken_manager.application.dto import CommandContext, RevokeProjectRoleCommand
from kraken_manager.application.errors import AuthorizationError, ConflictError, NotFoundError
from kraken_manager.domain.common import PrincipalId, ProjectId, validate_uuid
from kraken_manager.domain.identity import Principal, ProjectRole, SystemRole


def account_by_username(store: object, username: str) -> object:
    account = store.get_by_username(username)  # type: ignore[attr-defined]
    if account is None:
        raise ValueError(f"Учётная запись «{username}» не найдена")
    return account


def create_account(store: object, username: str, display_name: str, password: str) -> str:
    try:
        account = store.create_account(username, display_name, password, actor_id=None)  # type: ignore[attr-defined]
    except ValueError:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "IntegrityError":
            raise ValueError(f"Логин «{username}» уже занят") from exc
        raise
    return str(account.account_id)


def set_account_enabled(store: object, username: str, *, enabled: bool) -> str:
    account = account_by_username(store, username)
    try:
        updated = store.set_enabled(  # type: ignore[attr-defined]
            account.account_id,
            enabled,
            actor_id=None,
            preserve_last_admin=not enabled,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return str(updated.account_id)


def set_server_admin(store: object, username: str, *, grant: bool) -> str:
    account = account_by_username(store, username)
    try:
        if grant:
            store.grant_global_role(account.account_id, "server_admin", actor_id=None)  # type: ignore[attr-defined]
        else:
            store.revoke_global_role(  # type: ignore[attr-defined]
                account.account_id,
                "server_admin",
                actor_id=None,
                preserve_last_enabled=True,
            )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return str(account.account_id)


def reset_password(store: object, username: str, password: str) -> str:
    account = account_by_username(store, username)
    store.reset_password(account.account_id, password, actor_id=None)  # type: ignore[attr-defined]
    return str(account.account_id)


def revoke_sessions(store: object, username: str) -> str:
    account = account_by_username(store, username)
    store.revoke_all_sessions(account.account_id, actor_id=None)  # type: ignore[attr-defined]
    return str(account.account_id)


def revoke_maintainer(
    services: object,
    accounts: object,
    *,
    project: str,
    username: str,
) -> None:
    """Remove maintainer from any holder, including the last one in the project."""

    administrator = _local_administrator(accounts)
    target = account_by_username(accounts, username)
    project_id = _project_identifier(services, project)
    identity = services.identities.get(PrincipalId(target.account_id))  # type: ignore[attr-defined]
    if identity is None or not identity.active:
        raise ValueError(f"Участник «{username}» не найден в каталоге проектов")
    current = services.project_roles(project_id, str(target.account_id))  # type: ignore[attr-defined]
    actor = replace(
        Principal.local(
            subject=administrator.username,
            display_name=administrator.display_name,
            principal_id=administrator.account_id,
        ),
        system_roles=frozenset({SystemRole.SERVER_ADMIN}),
    )
    handler = RevokeProjectRoleHandler(services.uow_factory, services.profiles, services.clock)  # type: ignore[attr-defined]
    try:
        handler(
            RevokeProjectRoleCommand(
                context=CommandContext(actor=actor, idempotency_key=str(uuid4())),
                project_id=ProjectId(project_id),
                principal_id=PrincipalId(str(target.account_id)),
                role=ProjectRole.MAINTAINER,
                expected_revision=int(current["revision"]),
            )
        )
    except NotFoundError as exc:
        raise ValueError(str(exc)) from exc
    except ConflictError as exc:
        raise ValueError(str(exc)) from exc
    except AuthorizationError as exc:
        raise ValueError(str(exc)) from exc


def account_rows(store: object) -> list[dict[str, object]]:
    return [
        {
            "account_id": account.account_id,
            "username": account.username,
            "display_name": account.display_name,
            "enabled": bool(account.enabled),
            "system_roles": sorted(store.global_roles_for(account.account_id)),  # type: ignore[attr-defined]
            "created_at": str(account.created_at),
        }
        for account in store.list_accounts(include_disabled=True)  # type: ignore[attr-defined]
    ]


def audit_rows(store: object, *, limit: int = 500) -> list[dict[str, object]]:
    return [dict(item) for item in store.administration_audit(limit=limit)]  # type: ignore[attr-defined]


def project_rows(services: object) -> list[dict[str, object]]:
    return [
        {
            "project_id": str(item["project_id"]),
            "name": str(item["name"]),
            "state": str(item.get("state", "")),
        }
        for item in services.list_projects(include_archived=True)  # type: ignore[attr-defined]
    ]


def maintainer_rows(services: object, project_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for assignment in services.identities.assignments_for(ProjectId(project_id)):  # type: ignore[attr-defined]
        if assignment.role is not ProjectRole.MAINTAINER or not assignment.active:
            continue
        principal = services.identities.get(assignment.principal_id)  # type: ignore[attr-defined]
        rows.append(
            {
                "principal_id": str(assignment.principal_id),
                "username": principal.subject if principal is not None else str(assignment.principal_id),
                "display_name": principal.display_name if principal is not None else "",
            }
        )
    return rows


def connect_local_admin(config_path: Path) -> tuple[object, object]:
    from sqlalchemy import create_engine

    from kraken_manager.infrastructure.auth.local import Argon2PasswordHasher
    from kraken_manager.infrastructure.postgres.account_store import PostgresAccountStore
    from kraken_server.configuration import ServerConfig

    config = ServerConfig.load(config_path)
    engine = create_engine(config.database_url, pool_pre_ping=True)
    accounts = PostgresAccountStore(engine, Argon2PasswordHasher())
    return accounts, open_project_services(config.database_url, config.blob_root, engine=engine)


def open_project_services(database_url: str, blob_root: Path, *, engine: object | None = None) -> object:
    from kraken_manager.infrastructure.blob import FilesystemBlobStore
    from kraken_manager.infrastructure.postgres.unit_of_work import PostgresUnitOfWorkFactory
    from kraken_server.persistent_services import PostgresServerServices, ServerStorageProfiles

    if engine is None:
        from sqlalchemy import create_engine

        engine = create_engine(database_url, pool_pre_ping=True)
    blobs = FilesystemBlobStore(blob_root)
    services = PostgresServerServices(
        engine,
        PostgresUnitOfWorkFactory(engine, blobs),
        profiles=ServerStorageProfiles(),
    )
    services.clock = _Clock()
    return services


def _local_administrator(accounts: object) -> object:
    for account_id in accounts.accounts_with_global_role("server_admin"):  # type: ignore[attr-defined]
        account = accounts.get_account(account_id)  # type: ignore[attr-defined]
        if account is not None and account.enabled:
            return account
    raise ValueError("Нет включённого локального администратора")


def _project_identifier(services: object, project: str) -> str:
    from kraken_server.services import NotFoundError as MissingProject
    from kraken_server.services import ValidationError

    try:
        validate_uuid(project, field="project_id")
    except ValueError:
        matches = [
            item
            for item in services.list_projects(include_archived=True)  # type: ignore[attr-defined]
            if str(item.get("name")) == project
        ]
        if len(matches) != 1:
            raise ValueError(f"Проект «{project}» не найден")
        return str(matches[0]["project_id"])
    try:
        found = services.get_project(project)  # type: ignore[attr-defined]
    except (MissingProject, ValidationError) as exc:
        raise ValueError(f"Проект «{project}» не найден") from exc
    return str(found["project_id"])


class _Clock:
    def now(self):
        from datetime import UTC, datetime

        return datetime.now(UTC)
