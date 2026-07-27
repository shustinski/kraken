from .event_store import (
    CorruptEventLogError,
    EventStreamConflict,
    FilesystemEventStore,
    StoredEvent,
)
from .layout import FileProjectLayout, InvalidStorageIdentifier, validate_storage_identifier
from .locking import ProjectFileLock, ProjectLockTimeout
from .profile import (
    FILESYSTEM_CAPABILITIES,
    StorageCapabilities,
    StorageProfile,
    filesystem_storage_profile,
)
from .projection_store import ProjectionMutation, SQLiteProjectionStore

__all__ = [
    "CorruptEventLogError",
    "EventStreamConflict",
    "FILESYSTEM_CAPABILITIES",
    "FileProjectLayout",
    "FilesystemEventStore",
    "InvalidStorageIdentifier",
    "LocalProjectUnitOfWork",
    "LocalProjectUnitOfWorkFactory",
    "ProjectFileLock",
    "ProjectLockTimeout",
    "ProjectionMutation",
    "SQLiteProjectionStore",
    "StorageCapabilities",
    "StorageProfile",
    "StoredEvent",
    "filesystem_storage_profile",
    "validate_storage_identifier",
]


def __getattr__(name: str):
    # Avoid a package-initialisation cycle: the UoW composes the blob adapter,
    # while that adapter uses filesystem atomic-write helpers.
    if name in {"LocalProjectUnitOfWork", "LocalProjectUnitOfWorkFactory"}:
        from .unit_of_work import LocalProjectUnitOfWork, LocalProjectUnitOfWorkFactory

        return {
            "LocalProjectUnitOfWork": LocalProjectUnitOfWork,
            "LocalProjectUnitOfWorkFactory": LocalProjectUnitOfWorkFactory,
        }[name]
    raise AttributeError(name)
