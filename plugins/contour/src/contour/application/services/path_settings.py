from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..dto import PersistedPaths


class PathSettingsStore(Protocol):
    def load(self) -> PersistedPaths: ...

    def save(self, paths: PersistedPaths) -> None: ...


@dataclass(frozen=True, slots=True)
class DirectoryValidationResult:
    path: str
    exists: bool
    is_directory: bool

    @property
    def available(self) -> bool:
        return self.exists and self.is_directory


def normalize_path(path: str | Path) -> str:
    return str(Path(path))


def validate_existing_directory(path: str | Path) -> DirectoryValidationResult:
    normalized = normalize_path(path)
    candidate = Path(normalized)
    return DirectoryValidationResult(
        path=normalized,
        exists=candidate.exists(),
        is_directory=candidate.is_dir(),
    )


class PathSettingsController:
    def __init__(self, store: PathSettingsStore) -> None:
        self._store = store

    def load(self) -> PersistedPaths:
        return self._store.load()

    def save(self, paths: PersistedPaths) -> None:
        self._store.save(paths)

    def validate_input_directory(self, path: str | Path) -> DirectoryValidationResult:
        return validate_existing_directory(path)


__all__ = [
    "DirectoryValidationResult",
    "PathSettingsController",
    "PathSettingsStore",
    "normalize_path",
    "validate_existing_directory",
]
