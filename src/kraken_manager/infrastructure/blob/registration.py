from __future__ import annotations

from kraken_manager.infrastructure.storage_registry import BlobBackendRegistration

from .filesystem import FilesystemBlobStore


def blob_registration() -> BlobBackendRegistration:
    return BlobBackendRegistration(
        backend_id="filesystem",
        display_name="Filesystem SHA-256 objects",
        streaming=True,
        factory=FilesystemBlobStore,
    )


__all__ = ["blob_registration"]

