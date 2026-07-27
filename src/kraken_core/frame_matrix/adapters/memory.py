"""In-memory thumbnail store used for tests and non-persistent sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..storage import (
    InvalidationSelector,
    StoreCapability,
    StoreFull,
    StoreNamespace,
    StorePolicy,
    StoreStats,
    ThumbnailKey,
    ThumbnailRecord,
)


class MemoryThumbnailStore:
    capabilities = (
        StoreCapability.BATCH_READ
        | StoreCapability.BATCH_WRITE
        | StoreCapability.ATOMIC_REPLACE
        | StoreCapability.ACCESS_METADATA
        | StoreCapability.NAMESPACE_DELETE
    )

    def __init__(self) -> None:
        self._namespaces: dict[str, dict[ThumbnailKey, ThumbnailRecord]] = {}
        self._records: dict[ThumbnailKey, ThumbnailRecord] = {}
        self._namespace: StoreNamespace | None = None
        self._policy = StorePolicy()
        self._hits = self._misses = self._writes = self._errors = 0

    def open(self, namespace: StoreNamespace, policy: StorePolicy | None = None) -> None:
        self._namespace = namespace
        self._policy = policy or StorePolicy()
        self._records = self._namespaces.setdefault(namespace.digest(), {})

    def get(self, key: ThumbnailKey) -> ThumbnailRecord | None:
        record = self._records.get(key)
        if record is None:
            self._misses += 1
        else:
            self._hits += 1
        return record

    def get_many(self, keys: Iterable[ThumbnailKey]) -> Mapping[ThumbnailKey, ThumbnailRecord]:
        return {key: record for key in keys if (record := self.get(key)) is not None}

    def put(self, record: ThumbnailRecord) -> None:
        self.put_many((record,))

    def put_many(self, records: Iterable[ThumbnailRecord]) -> None:
        for record in records:
            if len(record.payload) > self._policy.max_entry_bytes:
                self._errors += 1
                raise StoreFull("thumbnail exceeds max_entry_bytes")
            self._records[record.key] = record
            self._writes += 1

    def delete(self, key: ThumbnailKey) -> bool:
        return self._records.pop(key, None) is not None

    def invalidate(self, selector: InvalidationSelector) -> int:
        keys = [key for key in self._records if selector.matches(key)]
        for key in keys:
            del self._records[key]
        return len(keys)

    def clear_namespace(self) -> None:
        self._records.clear()

    def stats(self) -> StoreStats:
        return StoreStats(
            entries=len(self._records),
            bytes=sum(len(record.payload) for record in self._records.values()),
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            errors=self._errors,
        )

    def compact(self) -> None:
        return None

    def close(self) -> None:
        return None


__all__ = ["MemoryThumbnailStore"]
