"""composition: frame cell snapshot."""
from __future__ import annotations
import os
from datetime import UTC, datetime
from pathlib import Path
from kraken_manager.application.ports import StorageProfile

def default_data_dir() -> Path:
    override = os.environ.get("KRAKEN_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".kraken").resolve()

class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

class DesktopStorageProfiles:
    def __init__(self, *profiles: StorageProfile) -> None:
        self._profiles = {profile.id: profile for profile in profiles}

    def get(self, profile_id: str) -> StorageProfile | None:
        return self._profiles.get(profile_id)

    def list(self) -> tuple[StorageProfile, ...]:
        return tuple(self._profiles.values())

# Compatibility exports: these values belong to the application boundary.
from kraken_manager.application.results import (  # noqa: E402,F401
    FrameCellSnapshot, FrameManagementState, IntegrityScanResult,
    ManagedImportResult, ProjectDeletionResult, UserSession as DesktopSession,
)
