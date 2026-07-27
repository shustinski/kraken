from __future__ import annotations

from kraken_manager.application.ports import StorageCapabilities
from kraken_manager.infrastructure.storage_registry import MetadataBackendRegistration

from .event_store import PostgresEventStore
from .projection_store import PostgresProjectionStore
from .unit_of_work import PostgresUnitOfWorkFactory


POSTGRES_CAPABILITIES = StorageCapabilities(
    multi_writer=True,
    transactions=True,
    snapshots=True,
    streaming=True,
    external_references=True,
    max_frames=1_000_000,
)


def backend_registration() -> MetadataBackendRegistration:
    return MetadataBackendRegistration(
        backend_id="postgresql",
        display_name="PostgreSQL",
        capabilities=POSTGRES_CAPABILITIES,
        event_store_factory=PostgresEventStore,
        projection_store_factory=PostgresProjectionStore,
        unit_of_work_factory=PostgresUnitOfWorkFactory,
    )


__all__ = ["POSTGRES_CAPABILITIES", "backend_registration"]
