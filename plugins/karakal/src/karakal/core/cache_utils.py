"""Small cache primitives shared by Karakal's disk and memory caches."""

from __future__ import annotations

import logging
import os
import pickle
import sys
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Generic, TypeVar

import numpy as np


_LOGGER = logging.getLogger(__name__)
_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class ByteLruCache(Generic[_KeyT, _ValueT]):
    """Thread-safe LRU constrained by retained bytes and an optional item cap."""

    def __init__(
        self,
        max_bytes: int,
        *,
        max_items: int | None = None,
        size_of: Callable[[_ValueT], int] | None = None,
    ) -> None:
        self._max_bytes = max(1, int(max_bytes))
        self._max_items = None if max_items is None else max(1, int(max_items))
        self._size_of = size_of or estimate_size_bytes
        self._items: OrderedDict[_KeyT, tuple[_ValueT, int]] = OrderedDict()
        self._total_bytes = 0
        self._lock = RLock()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def set_max_bytes(self, max_bytes: int) -> None:
        with self._lock:
            self._max_bytes = max(1, int(max_bytes))
            self._trim()

    def get(self, key: _KeyT) -> _ValueT | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            self._items.move_to_end(key)
            return entry[0]

    def put(self, key: _KeyT, value: _ValueT) -> None:
        retained = max(0, int(self._size_of(value)))
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._total_bytes -= previous[1]
            self._items[key] = (value, retained)
            self._total_bytes += retained
            self._trim()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._total_bytes = 0

    def _trim(self) -> None:
        while self._items and (
            self._total_bytes > self._max_bytes or (self._max_items is not None and len(self._items) > self._max_items)
        ):
            _key, (_value, retained) = self._items.popitem(last=False)
            self._total_bytes = max(0, self._total_bytes - retained)


def estimate_size_bytes(value: object, *, _seen: set[int] | None = None) -> int:
    """Estimate retained bytes without walking the same object more than once."""

    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    size = int(sys.getsizeof(value, 0))
    if isinstance(value, Mapping):
        return size + sum(
            estimate_size_bytes(key, _seen=seen) + estimate_size_bytes(item, _seen=seen) for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return size + sum(estimate_size_bytes(item, _seen=seen) for item in value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        size += estimate_size_bytes(attributes, _seen=seen)
    for slot in getattr(type(value), "__slots__", ()):
        if isinstance(slot, str) and hasattr(value, slot):
            size += estimate_size_bytes(getattr(value, slot), _seen=seen)
    return size


def atomic_pickle_dump(path: Path, payload: object) -> None:
    """Serialize, verify readability, and atomically replace one cache entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as handle:
            pickle.load(handle)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            _LOGGER.debug("Could not remove temporary cache file %s: %s", temporary, error)


def trim_directory_by_bytes(
    directory: Path,
    *,
    max_bytes: int,
    pattern: str = "*.pickle",
    max_files: int | None = None,
) -> tuple[int, int]:
    """Delete oldest cache entries until both byte and optional count limits hold."""

    try:
        entries = [(path, path.stat()) for path in directory.glob(pattern) if path.is_file()]
    except OSError as error:
        _LOGGER.warning("Could not inspect cache directory %s: %s", directory, error)
        return 0, 0
    entries.sort(key=lambda item: item[1].st_mtime_ns)
    total_bytes = sum(int(stat.st_size) for _path, stat in entries)
    removed_files = 0
    removed_bytes = 0
    file_limit = len(entries) if max_files is None else max(0, int(max_files))
    byte_limit = max(0, int(max_bytes))
    while entries and (total_bytes > byte_limit or len(entries) > file_limit):
        path, stat = entries.pop(0)
        try:
            path.unlink()
        except OSError as error:
            _LOGGER.warning("Could not trim cache entry %s: %s", path, error)
            continue
        size = int(stat.st_size)
        total_bytes -= size
        removed_files += 1
        removed_bytes += size
    return removed_files, removed_bytes


__all__ = ["ByteLruCache", "atomic_pickle_dump", "estimate_size_bytes", "trim_directory_by_bytes"]
