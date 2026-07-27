from __future__ import annotations

from kraken_manager.infrastructure.storage_registry import MetadataBackendRegistration

from .event_store import FilesystemEventStore
from .profile import FILESYSTEM_CAPABILITIES
from .projection_store import SQLiteProjectionStore
from .unit_of_work import LocalProjectUnitOfWorkFactory


def backend_registration() -> MetadataBackendRegistration:
    return MetadataBackendRegistration(
        backend_id="filesystem",
        display_name="Local filesystem",
        capabilities=FILESYSTEM_CAPABILITIES,
        event_store_factory=FilesystemEventStore,
        projection_store_factory=SQLiteProjectionStore,
        unit_of_work_factory=LocalProjectUnitOfWorkFactory,
    )


__all__ = ["backend_registration"]

