from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..dto import PersistedPaths


class PathSettingsStore(Protocol):
    def load(self) -> PersistedPaths: ...

    def save(self, paths: PersistedPaths) -> None: ...


@runtime_checkable
class RecentPathsStore(Protocol):
    def load_recent_paths(self, role: str) -> list[str]: ...

    def save_recent_paths(self, role: str, paths: list[str]) -> None: ...


MAX_RECENT_PATHS = 8


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


def merge_recent_paths(existing: list[str], path: str | Path, *, limit: int = MAX_RECENT_PATHS) -> list[str]:
    """Put ``path`` first, drop duplicates (case-insensitive on Windows-like paths)."""

    text = str(path or "").strip()
    if not text:
        return list(existing)[:limit]
    newest = normalize_path(text)
    seen = {newest.casefold()}
    merged = [newest]
    for entry in existing:
        candidate = str(entry or "").strip()
        if not candidate:
            continue
        normalized = normalize_path(candidate)
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged[:limit]


def validate_existing_directory(path: str | Path) -> DirectoryValidationResult:
    normalized = normalize_path(path)
    candidate = Path(normalized)
    return DirectoryValidationResult(
        path=normalized,
        exists=candidate.exists(),
        is_directory=candidate.is_dir(),
    )


class PathSettingsController:
    def __init__(self, store: PathSettingsStore, recent_store: RecentPathsStore | None = None) -> None:
        self._store = store
        if recent_store is None and isinstance(store, RecentPathsStore):
            recent_store = store
        self._recent_store = recent_store

    def load(self) -> PersistedPaths:
        return self._store.load()

    def save(self, paths: PersistedPaths) -> None:
        self._store.save(paths)

    def validate_input_directory(self, path: str | Path) -> DirectoryValidationResult:
        return validate_existing_directory(path)

    def recent_paths(self, role: str) -> list[str]:
        if self._recent_store is None:
            return []
        return list(self._recent_store.load_recent_paths(role))[:MAX_RECENT_PATHS]

    def remember_recent_path(self, role: str, path: str | Path) -> None:
        if self._recent_store is None or not str(path or "").strip():
            return
        merged = merge_recent_paths(self.recent_paths(role), path)
        self._recent_store.save_recent_paths(role, merged)


__all__ = [
    "MAX_RECENT_PATHS",
    "DirectoryValidationResult",
    "PathSettingsController",
    "PathSettingsStore",
    "RecentPathsStore",
    "merge_recent_paths",
    "normalize_path",
    "validate_existing_directory",
]
