"""Entry-point discovery for semantic metadata/blob backend adapters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable

from kraken_manager.application.ports import StorageCapabilities


@dataclass(frozen=True, slots=True)
class MetadataBackendRegistration:
    backend_id: str
    display_name: str
    capabilities: StorageCapabilities
    event_store_factory: Callable[..., Any]
    projection_store_factory: Callable[..., Any] | None = None
    unit_of_work_factory: Callable[..., Any] | None = None


@dataclass(frozen=True, slots=True)
class BlobBackendRegistration:
    backend_id: str
    display_name: str
    streaming: bool
    factory: Callable[..., Any]


def _discover(group: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in metadata.entry_points().select(group=group):
        loaded = entry.load()
        registration = loaded() if callable(loaded) else loaded
        if entry.name in result:
            raise RuntimeError(f"Duplicate Kraken backend entry point: {group}/{entry.name}")
        result[entry.name] = registration
    return result


def discover_metadata_backends() -> dict[str, MetadataBackendRegistration]:
    discovered = _discover("kraken.storage_backends")
    for name, value in discovered.items():
        if not isinstance(value, MetadataBackendRegistration) or value.backend_id != name:
            raise RuntimeError(f"Invalid metadata backend registration: {name}")
    return discovered


def discover_blob_backends() -> dict[str, BlobBackendRegistration]:
    discovered = _discover("kraken.blob_backends")
    for name, value in discovered.items():
        if not isinstance(value, BlobBackendRegistration) or value.backend_id != name:
            raise RuntimeError(f"Invalid blob backend registration: {name}")
    return discovered


__all__ = [
    "BlobBackendRegistration",
    "MetadataBackendRegistration",
    "discover_blob_backends",
    "discover_metadata_backends",
]

