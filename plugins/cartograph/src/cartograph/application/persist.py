"""Persistence use case wrapping the JSON local-block store."""

from __future__ import annotations

from pathlib import Path

from cartograph.domain.topology import LocalBlockSolution
from cartograph.infrastructure.persistence import JsonLocalBlockStore


class PersistLocalBlock:
    def __init__(self, root: Path) -> None:
        self._store = JsonLocalBlockStore(root)

    def save(self, key: str, solution: LocalBlockSolution) -> Path:
        self._store.save(key, solution)
        return self._store.path_for(key)

    def load(self, key: str) -> LocalBlockSolution | None:
        return self._store.load(key)
