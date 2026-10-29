"""Open the local PostgreSQL stores the administration window and commands use."""

from __future__ import annotations

from pathlib import Path


def connect_local_admin(config_path: Path) -> tuple[object, object]:
    from sqlalchemy import create_engine

    from kraken_manager.infrastructure.auth.local import Argon2PasswordHasher
    from kraken_manager.infrastructure.postgres.account_store import PostgresAccountStore
    from kraken_server.configuration import ServerConfig

    config = ServerConfig.load(config_path)
    # An unavailable database must not leave the administration window waiting
    # indefinitely before it can report the connection failure.
    engine = create_engine(config.database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    accounts = PostgresAccountStore(engine, Argon2PasswordHasher())
    services = open_project_services(config.database_url, config.blob_root, engine=engine)
    from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry

    services.workspace_files = WorkspaceFileService(WorkspaceRegistry(config.blob_root / "workspaces"))
    services.source_root = str(config.source_root)
    services.derived_root = str(config.derived_root)
    return accounts, services


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


class _Clock:
    def now(self):
        from datetime import UTC, datetime

        return datetime.now(UTC)
