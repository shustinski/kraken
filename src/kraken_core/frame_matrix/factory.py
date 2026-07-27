"""Lazy registry and URI factory for thumbnail-store adapters."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

from .storage import StoreUnavailable, ThumbnailStore


StoreBuilder = Callable[[str], ThumbnailStore]


class ThumbnailStoreFactory:
    ENTRY_POINT_GROUP = "kraken.thumbnail_stores"

    def __init__(self) -> None:
        self._builders: dict[str, StoreBuilder] = {}
        self._entry_points_loaded = False
        self.register("memory", self._build_memory)
        self.register("files", self._build_files)
        self.register("sqlite", self._build_sqlite)

    def register(self, scheme: str, builder: StoreBuilder, *, replace: bool = False) -> None:
        normalized = str(scheme).strip().lower()
        if not normalized or any(character in normalized for character in ":/"):
            raise ValueError("invalid thumbnail store scheme")
        if normalized in self._builders and not replace:
            raise ValueError(f"thumbnail store scheme {normalized!r} is already registered")
        self._builders[normalized] = builder

    def create(self, uri: str) -> ThumbnailStore:
        parsed = urlparse(str(uri))
        scheme = parsed.scheme.lower()
        if not scheme:
            raise StoreUnavailable("thumbnail store URI must include a scheme")
        self._load_entry_points()
        builder = self._builders.get(scheme)
        if builder is None:
            raise StoreUnavailable(f"unsupported thumbnail store scheme: {scheme}")
        location = self._location(parsed)
        try:
            return builder(location)
        except StoreUnavailable:
            raise
        except Exception as exc:
            raise StoreUnavailable(f"could not create {scheme} thumbnail store: {exc}") from exc

    def schemes(self) -> tuple[str, ...]:
        self._load_entry_points()
        return tuple(sorted(self._builders))

    def _load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        try:
            discovered = metadata.entry_points(group=self.ENTRY_POINT_GROUP)
        except TypeError:  # pragma: no cover - compatibility with older importlib metadata
            discovered = metadata.entry_points().select(group=self.ENTRY_POINT_GROUP)
        for entry_point in discovered:
            if entry_point.name.lower() in self._builders:
                continue
            builder = entry_point.load()
            if callable(builder):
                self._builders[entry_point.name.lower()] = builder

    @staticmethod
    def _location(parsed) -> str:
        if parsed.scheme == "memory":
            return ""
        if parsed.netloc:
            location = unquote(f"//{parsed.netloc}{parsed.path}")
        else:
            location = unquote(parsed.path)
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", location):
            location = location[1:]
        if not location:
            raise StoreUnavailable(f"{parsed.scheme} thumbnail store requires a path")
        return location

    @staticmethod
    def _build_memory(_location: str) -> ThumbnailStore:
        from .adapters.memory import MemoryThumbnailStore

        return MemoryThumbnailStore()

    @staticmethod
    def _build_files(location: str) -> ThumbnailStore:
        from .adapters.filesystem import FilesystemThumbnailStore

        return FilesystemThumbnailStore(Path(location))

    @staticmethod
    def _build_sqlite(location: str) -> ThumbnailStore:
        from .adapters.sqlite import ShardedSQLiteThumbnailStore

        return ShardedSQLiteThumbnailStore(Path(location))


__all__ = ["ThumbnailStoreFactory"]
