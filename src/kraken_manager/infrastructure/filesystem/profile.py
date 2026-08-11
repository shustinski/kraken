from __future__ import annotations

from kraken_manager.application.dto import StorageBackendKind, StorageScope
from kraken_manager.application.ports import StorageCapabilities, StorageProfile


FILESYSTEM_CAPABILITIES = StorageCapabilities(
    multi_writer=False,
    transactions=True,
    snapshots=True,
    streaming=True,
    external_references=True,
    max_frames=None,
)


def filesystem_storage_profile(
    root: str | None = None,
    *,
    profile_id: str = "local-filesystem",
    display_name: str = "Local filesystem",
) -> StorageProfile:
    """Return the application-facing local profile.

    ``root`` is accepted for composition-root convenience, but paths deliberately
    do not become part of the public profile or leak into the domain model.
    """

    del root
    return StorageProfile(
        id=profile_id,
        name=display_name,
        metadata_backend=StorageBackendKind.FILESYSTEM,
        blob_backend="filesystem-sha256-v1",
        scope=StorageScope.LOCAL,
        capabilities=FILESYSTEM_CAPABILITIES,
    )


__all__ = [
    "FILESYSTEM_CAPABILITIES",
    "StorageCapabilities",
    "StorageProfile",
    "filesystem_storage_profile",
]
